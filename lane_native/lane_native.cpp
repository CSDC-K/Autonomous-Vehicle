/*
 * lane_native — C++ accelerated lane detection for lane_opencv.py
 * ─────────────────────────────────────────────────────────────────
 * pybind11 module exporting process_frame() that replaces the hot
 * Python path: detect_lines → separate → cluster → fit → classify.
 *
 * Build:  cd lane_native && pip install .
 * Usage:  import lane_native
 *         result = lane_native.process_frame(frame, params)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <tuple>
#include <optional>
#include <numeric>

namespace py = pybind11;

// ── Segment type ────────────────────────────────────────────────
struct Seg { int x1, y1, x2, y2; };

// ── Helper: numpy ndarray → cv::Mat (zero-copy) ────────────────
static cv::Mat ndarray_to_mat(py::array_t<uint8_t>& arr) {
    auto buf = arr.request();
    if (buf.ndim == 3) {
        int h = buf.shape[0], w = buf.shape[1], c = buf.shape[2];
        return cv::Mat(h, w, CV_8UC(c), buf.ptr);
    } else if (buf.ndim == 2) {
        int h = buf.shape[0], w = buf.shape[1];
        return cv::Mat(h, w, CV_8UC1, buf.ptr);
    }
    throw std::runtime_error("ndarray_to_mat: unsupported ndim");
}

// ── Helper: cv::Mat → numpy ndarray (copy) ─────────────────────
static py::array_t<uint8_t> mat_to_ndarray(const cv::Mat& m) {
    if (m.empty()) return py::none().cast<py::array_t<uint8_t>>();
    std::vector<ssize_t> shape, strides;
    if (m.channels() == 1) {
        shape   = {m.rows, m.cols};
        strides = {(ssize_t)m.step[0], (ssize_t)m.step[1]};
    } else {
        shape   = {m.rows, m.cols, m.channels()};
        strides = {(ssize_t)m.step[0], (ssize_t)m.step[1], (ssize_t)m.elemSize1()};
    }
    auto arr = py::array_t<uint8_t>(shape, strides);
    std::memcpy(arr.mutable_data(), m.data, m.total() * m.elemSize());
    return arr;
}

// ╔══════════════════════════════════════════════════════════════╗
// ║                   DETECT LINES                              ║
// ╚══════════════════════════════════════════════════════════════╝

static void detect_lines(
    const cv::Mat& frame,
    int bright_thr,
    bool light_line_on_dark,
    int blur_kernel,
    int canny_low, int canny_high,
    int hough_threshold, int hough_min_len, int hough_max_gap,
    int min_line_angle,
    int min_contour_area, double min_contour_aspect,
    double roi_top,
    bool build_viz,
    /* outputs */
    std::vector<Seg>& out_lines,
    cv::Mat& out_viz
) {
    int h = frame.rows, w = frame.cols;

    // ROI mask
    cv::Mat roi_mask = cv::Mat::zeros(h, w, CV_8UC1);
    int y_top = (int)(h * roi_top);
    cv::Point pts[4] = {
        {0, h}, {0, y_top}, {w, y_top}, {w, h}
    };
    cv::fillConvexPoly(roi_mask, pts, 4, cv::Scalar(255));

    // Gray + Blur
    cv::Mat gray, blur;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(blur_kernel, blur_kernel), 0);

    // Threshold
    cv::Mat binary;
    int thr_type = light_line_on_dark ? cv::THRESH_BINARY : cv::THRESH_BINARY_INV;
    cv::threshold(blur, binary, bright_thr, 255, thr_type);

    // Apply ROI
    cv::Mat masked;
    cv::bitwise_and(binary, roi_mask, masked);

    // Morphology
    cv::Mat k3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat k5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 5));
    cv::morphologyEx(masked, masked, cv::MORPH_CLOSE, k3);
    cv::dilate(masked, masked, k5, cv::Point(-1,-1), 1);

    // Contour filtering (glare removal)
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(masked, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat clean = cv::Mat::zeros(h, w, CV_8UC1);
    for (auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < min_contour_area) continue;
        cv::RotatedRect rect = cv::minAreaRect(c);
        float w_r = rect.size.width, h_r = rect.size.height;
        double aspect = std::max(w_r, h_r) / (std::min(w_r, h_r) + 1e-3);
        if (aspect < min_contour_aspect) continue;
        cv::drawContours(clean, std::vector<std::vector<cv::Point>>{c}, -1,
                         cv::Scalar(255), cv::FILLED);
    }
    masked = clean;

    // Canny
    cv::Mat edges;
    cv::Canny(masked, edges, canny_low, canny_high);

    // HoughLinesP
    std::vector<cv::Vec4i> raw;
    cv::HoughLinesP(edges, raw, 1.0, CV_PI / 180.0,
                    hough_threshold, hough_min_len, hough_max_gap);

    // Angle filter
    out_lines.clear();
    for (auto& l : raw) {
        double angle = std::abs(std::atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI);
        if (angle >= min_line_angle && angle <= (180 - min_line_angle)) {
            out_lines.push_back({l[0], l[1], l[2], l[3]});
        }
    }

    // Viz
    if (build_viz) {
        cv::cvtColor(masked, out_viz, cv::COLOR_GRAY2BGR);
        for (auto& s : out_lines) {
            cv::line(out_viz, {s.x1, s.y1}, {s.x2, s.y2}, {0, 255, 0}, 1);
        }
    } else {
        out_viz = cv::Mat();
    }
}

// ╔══════════════════════════════════════════════════════════════╗
// ║                   SEPARATE LINES                            ║
// ╚══════════════════════════════════════════════════════════════╝

static void separate_lines(
    const std::vector<Seg>& lines, int w,
    std::vector<Seg>& left, std::vector<Seg>& right
) {
    double mid = w / 2.0;
    left.clear(); right.clear();
    for (auto& s : lines) {
        double slope = (s.y2 - s.y1) / ((double)(s.x2 - s.x1) + 1e-6);
        if (std::abs(slope) < 0.2) continue;
        double cx = (s.x1 + s.x2) / 2.0;
        if (cx < mid) left.push_back(s);
        else          right.push_back(s);
    }
}

// ╔══════════════════════════════════════════════════════════════╗
// ║              PICK INNER SEGMENTS (CLUSTER)                  ║
// ╚══════════════════════════════════════════════════════════════╝

static int seg_bottom_x(const Seg& s) {
    return (s.y2 > s.y1) ? s.x2 : s.x1;
}

static std::vector<Seg> pick_inner_segments(
    const std::vector<Seg>& segments,
    bool is_left, int w, int cluster_gap
) {
    if (segments.empty()) return {};

    // Tag with bottom_x and sort
    struct Tagged { Seg s; int bx; };
    std::vector<Tagged> tagged;
    tagged.reserve(segments.size());
    for (auto& seg : segments)
        tagged.push_back({seg, seg_bottom_x(seg)});
    std::sort(tagged.begin(), tagged.end(),
              [](const Tagged& a, const Tagged& b) { return a.bx < b.bx; });

    // Build clusters
    std::vector<std::vector<Tagged>> clusters;
    clusters.push_back({tagged[0]});
    for (size_t i = 1; i < tagged.size(); i++) {
        if (tagged[i].bx - tagged[i-1].bx > cluster_gap)
            clusters.emplace_back();
        clusters.back().push_back(tagged[i]);
    }

    double mid = w / 2.0;
    const std::vector<Tagged>* best = nullptr;
    double best_dist = 1e18;

    // Pass 1: correct-side clusters
    for (auto& c : clusters) {
        double cx = 0;
        for (auto& t : c) cx += t.bx;
        cx /= c.size();
        double dist = is_left ? (mid - cx) : (cx - mid);
        if (dist >= 0 && dist < best_dist) {
            best_dist = dist;
            best = &c;
        }
    }

    // Pass 2: fallback — closest cluster
    if (!best) {
        for (auto& c : clusters) {
            double cx = 0;
            for (auto& t : c) cx += t.bx;
            cx /= c.size();
            double dist = std::abs(cx - mid);
            if (dist < best_dist) {
                best_dist = dist;
                best = &c;
            }
        }
    }

    if (!best) return segments; // shouldn't happen

    std::vector<Seg> result;
    result.reserve(best->size());
    for (auto& t : *best) result.push_back(t.s);
    return result;
}

// ╔══════════════════════════════════════════════════════════════╗
// ║                      FIT LINE                               ║
// ╚══════════════════════════════════════════════════════════════╝

struct FitResult { int x1, y1, x2, y2; };

static std::optional<FitResult> fit_line(
    const std::vector<Seg>& segments, int h, double roi_top
) {
    if (segments.empty()) return std::nullopt;

    // Collect xs, ys
    std::vector<double> xs, ys;
    xs.reserve(segments.size() * 2);
    ys.reserve(segments.size() * 2);
    for (auto& s : segments) {
        xs.push_back(s.x1); ys.push_back(s.y1);
        xs.push_back(s.x2); ys.push_back(s.y2);
    }
    if (xs.size() < 2) return std::nullopt;

    // Simple linear regression: x = a*y + b  (polyfit(ys, xs, 1))
    int n = (int)xs.size();
    double sum_y = 0, sum_x = 0, sum_yy = 0, sum_yx = 0;
    for (int i = 0; i < n; i++) {
        sum_y  += ys[i];
        sum_x  += xs[i];
        sum_yy += ys[i] * ys[i];
        sum_yx += ys[i] * xs[i];
    }
    double denom = n * sum_yy - sum_y * sum_y;
    if (std::abs(denom) < 1e-12) return std::nullopt;

    double a = (n * sum_yx - sum_y * sum_x) / denom;
    double b = (sum_x - a * sum_y) / n;

    int y_bot = h;
    int y_top = (int)(h * roi_top);
    return FitResult{
        (int)(a * y_top + b), y_top,
        (int)(a * y_bot + b), y_bot
    };
}

// ╔══════════════════════════════════════════════════════════════╗
// ║                  GET LANE X AT Y                            ║
// ╚══════════════════════════════════════════════════════════════╝

static std::optional<double> get_lane_x_at_y(
    const std::vector<Seg>& segments, int target_y
) {
    if (segments.empty()) return std::nullopt;

    std::vector<double> x_vals;
    x_vals.reserve(segments.size());
    for (auto& s : segments) {
        int dy = s.y2 - s.y1;
        if (std::abs(dy) < 1) {
            x_vals.push_back((s.x1 + s.x2) / 2.0);
            continue;
        }
        double t = (target_y - s.y1) / (double)dy;
        x_vals.push_back(s.x1 + t * (s.x2 - s.x1));
    }
    if (x_vals.empty()) return std::nullopt;

    // Median
    std::sort(x_vals.begin(), x_vals.end());
    size_t mid = x_vals.size() / 2;
    if (x_vals.size() % 2 == 0)
        return (x_vals[mid - 1] + x_vals[mid]) / 2.0;
    return x_vals[mid];
}

// ╔══════════════════════════════════════════════════════════════╗
// ║              CLASSIFY LINE TYPE (DASHED/SOLID)              ║
// ╚══════════════════════════════════════════════════════════════╝

static int count_gaps(
    const std::vector<Seg>& segments,
    const std::optional<FitResult>& fitted,
    int dashed_gap_px
) {
    if (segments.size() < 2 || !fitted) return 0;

    // Project segments onto y-axis
    std::vector<std::pair<int,int>> projs;
    projs.reserve(segments.size());
    for (auto& s : segments) {
        projs.emplace_back(std::min(s.y1, s.y2), std::max(s.y1, s.y2));
    }
    std::sort(projs.begin(), projs.end());

    // Merge overlapping
    std::vector<std::pair<int,int>> merged;
    merged.push_back(projs[0]);
    for (size_t i = 1; i < projs.size(); i++) {
        auto& prev = merged.back();
        if (projs[i].first <= prev.second + (int)(dashed_gap_px * 0.5)) {
            prev.second = std::max(prev.second, projs[i].second);
        } else {
            merged.push_back(projs[i]);
        }
    }

    int gaps = 0;
    for (size_t i = 1; i < merged.size(); i++) {
        if (merged[i].first - merged[i-1].second >= dashed_gap_px)
            gaps++;
    }
    return gaps;
}

static std::string classify_line_type(
    const std::vector<Seg>& segments,
    const std::optional<FitResult>& fitted,
    double dashed_coverage_thr,
    int dashed_min_gaps,
    int dashed_gap_px
) {
    if (segments.empty() || !fitted) return "none";

    double total_seg_len = 0;
    for (auto& s : segments) {
        double dx = s.x2 - s.x1, dy = s.y2 - s.y1;
        total_seg_len += std::sqrt(dx*dx + dy*dy);
    }

    int line_span = std::abs(fitted->y2 - fitted->y1);
    if (line_span < 10) return "none";

    double coverage = total_seg_len / line_span;
    int gaps = count_gaps(segments, fitted, dashed_gap_px);

    if (coverage < dashed_coverage_thr && gaps >= dashed_min_gaps) return "dashed";
    if (coverage < dashed_coverage_thr) return "dashed";
    if (gaps >= dashed_min_gaps + 1) return "dashed";
    return "solid";
}

// ╔══════════════════════════════════════════════════════════════╗
// ║                 MAIN: process_frame                         ║
// ╚══════════════════════════════════════════════════════════════╝

static py::dict process_frame(
    py::array_t<uint8_t> frame_arr,
    int bright_thr,
    bool build_viz,
    /* calibration params (passed from Python) */
    bool   light_line_on_dark,
    int    blur_kernel,
    int    canny_low,
    int    canny_high,
    int    hough_threshold,
    int    hough_min_len,
    int    hough_max_gap,
    int    min_line_angle,
    int    min_contour_area,
    double min_contour_aspect,
    double roi_top,
    int    lane_cluster_gap,
    double measurement_y_ratio,
    double dashed_coverage_thr,
    int    dashed_min_gaps,
    int    dashed_gap_px
) {
    cv::Mat frame = ndarray_to_mat(frame_arr);
    int h = frame.rows, w = frame.cols;

    // 1) Detect lines
    std::vector<Seg> lines;
    cv::Mat viz;
    detect_lines(frame, bright_thr, light_line_on_dark,
                 blur_kernel, canny_low, canny_high,
                 hough_threshold, hough_min_len, hough_max_gap,
                 min_line_angle, min_contour_area, min_contour_aspect,
                 roi_top, build_viz, lines, viz);

    // 2) Separate left/right
    std::vector<Seg> left_segs, right_segs;
    separate_lines(lines, w, left_segs, right_segs);

    // 3) Pick inner segments
    auto left_inner  = pick_inner_segments(left_segs,  true,  w, lane_cluster_gap);
    auto right_inner = pick_inner_segments(right_segs, false, w, lane_cluster_gap);

    // 4) Fit lines
    auto left_fit  = fit_line(left_inner,  h, roi_top);
    auto right_fit = fit_line(right_inner, h, roi_top);

    // 5) Get lane x at measurement y
    int measurement_y = (int)(h * measurement_y_ratio);
    auto left_x  = get_lane_x_at_y(left_inner,  measurement_y);
    auto right_x = get_lane_x_at_y(right_inner, measurement_y);

    // 6) Classify line types
    auto left_type  = classify_line_type(left_inner,  left_fit,
                                         dashed_coverage_thr, dashed_min_gaps, dashed_gap_px);
    auto right_type = classify_line_type(right_inner, right_fit,
                                         dashed_coverage_thr, dashed_min_gaps, dashed_gap_px);

    // ── Build result dict ──
    py::dict result;

    // Convert segments to Python lists of tuples
    auto segs_to_list = [](const std::vector<Seg>& segs) {
        py::list lst;
        for (auto& s : segs)
            lst.append(py::make_tuple(s.x1, s.y1, s.x2, s.y2));
        return lst;
    };

    result["left_segs"]  = segs_to_list(left_inner);
    result["right_segs"] = segs_to_list(right_inner);

    if (left_fit)
        result["left_line"] = py::make_tuple(left_fit->x1, left_fit->y1,
                                              left_fit->x2, left_fit->y2);
    else
        result["left_line"] = py::none();

    if (right_fit)
        result["right_line"] = py::make_tuple(right_fit->x1, right_fit->y1,
                                               right_fit->x2, right_fit->y2);
    else
        result["right_line"] = py::none();

    if (left_x)
        result["left_x"] = *left_x;
    else
        result["left_x"] = py::none();

    if (right_x)
        result["right_x"] = *right_x;
    else
        result["right_x"] = py::none();

    result["left_type"]  = left_type;
    result["right_type"] = right_type;

    if (build_viz && !viz.empty())
        result["viz_mask"] = mat_to_ndarray(viz);
    else
        result["viz_mask"] = py::none();

    return result;
}

// ╔══════════════════════════════════════════════════════════════╗
// ║                    PYBIND11 MODULE                          ║
// ╚══════════════════════════════════════════════════════════════╝

PYBIND11_MODULE(lane_native, m) {
    m.doc() = "C++ accelerated lane detection for lane_opencv.py";

    m.def("process_frame", &process_frame,
        py::arg("frame"),
        py::arg("bright_thr"),
        py::arg("build_viz"),
        py::arg("light_line_on_dark"),
        py::arg("blur_kernel"),
        py::arg("canny_low"),
        py::arg("canny_high"),
        py::arg("hough_threshold"),
        py::arg("hough_min_len"),
        py::arg("hough_max_gap"),
        py::arg("min_line_angle"),
        py::arg("min_contour_area"),
        py::arg("min_contour_aspect"),
        py::arg("roi_top"),
        py::arg("lane_cluster_gap"),
        py::arg("measurement_y_ratio"),
        py::arg("dashed_coverage_thr"),
        py::arg("dashed_min_gaps"),
        py::arg("dashed_gap_px"),
        "Process a single BGR frame through the full lane detection pipeline.\n"
        "Returns dict with keys: left_segs, right_segs, left_line, right_line,\n"
        "left_x, right_x, left_type, right_type, viz_mask"
    );
}
