"""
AI Model for Hailo Pose Estimation - Picamera2 + GStreamer Hailo Pipeline
Uses Picamera2 for camera capture and GStreamer for Hailo inference
"""
import sys
import os
import threading
import time
import numpy as np
import cv2
from collections import deque

# --- Picamera2 Setup ---
try:
    from picamera2 import Picamera2
except ImportError:
    print("HATA: Picamera2 bulunamadı!")
    Picamera2 = None

# --- GStreamer & Hailo Setup ---
try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import Gst, GLib, GstApp
    import hailo
except ImportError as e:
    print(f"HATA: Gerekli modüller eksik. {e}")
    sys.exit(1)

# Initialize GStreamer once
Gst.init(None)

# --- CONFIGURATION ---
TARGET_REPS = 5
FRAME_W, FRAME_H = 640, 640
SMOOTHING_WINDOW = 10
CAMERA_W, CAMERA_H = 640, 480

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Use the official Hailo model file from system
MY_HEF_PATH = "/usr/share/hailo-models/yolov8s_pose_h8l_pi.hef"
MY_POST_SO = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolov8pose_post.so"
MY_POST_FUNC = "filter"

# --- GLOBAL STATE ---
_lock = threading.Lock()
_pipeline = None
_main_loop = None
_loop_thread = None
_camera_thread = None
_picam2 = None
_is_running = False
_stop_flag = False
_latest_frame_data = None
_squat_logic = None


class SquatLogic:
    """Handles squat counting logic with angle-based detection"""
    
    def __init__(self):
        self.counter = 0
        self.stage = "calibrating"
        self.finished = False
        self.angle_history = deque(maxlen=SMOOTHING_WINDOW)
        self.ref_hip_y = 0
        self.ref_leg_len = 0
        self.calibration_frames = 0
        self.down_frames = 0
        self.up_frames = 0
        self.MIN_CALIBRATION_FRAMES = 15
        self.MIN_DOWN_FRAMES = 5
        self.MIN_UP_FRAMES = 5

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 180.0
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def process(self, l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle):
        """Process squat using both legs, take minimum angle (deeper squat)"""
        # Calculate angle for both legs
        left_angle = self.calculate_angle(l_hip, l_knee, l_ankle)
        right_angle = self.calculate_angle(r_hip, r_knee, r_ankle)
        
        # Handle NaN
        if np.isnan(left_angle):
            left_angle = 180.0
        if np.isnan(right_angle):
            right_angle = 180.0
        
        # Use minimum angle (deeper squat) for better detection
        raw_angle = min(left_angle, right_angle)

        self.angle_history.append(raw_angle)
        avg_angle = int(sum(self.angle_history) / len(self.angle_history))

        # Use average hip position
        current_hip_y = (l_hip[1] + r_hip[1]) / 2
        current_leg_len = abs((l_ankle[1] + r_ankle[1]) / 2 - current_hip_y)

        debug_msg = "WAIT"
        debug_color = (255, 255, 255)

        if self.stage == "calibrating":
            debug_msg = "DIK DUR"
            debug_color = (0, 165, 255)
            if avg_angle > 165:
                self.calibration_frames += 1
                if self.calibration_frames >= self.MIN_CALIBRATION_FRAMES:
                    self.ref_hip_y = current_hip_y
                    self.ref_leg_len = current_leg_len
                    self.stage = "up"
                    self.up_frames = 0
            else:
                self.calibration_frames = 0

        elif self.stage == "up":
            debug_msg = "SQUAT YAP"
            debug_color = (0, 255, 0)
            
            if current_hip_y < self.ref_hip_y:
                self.ref_hip_y = current_hip_y
            if current_leg_len > self.ref_leg_len:
                self.ref_leg_len = current_leg_len
            
            # Count squat when knee angle goes below 162 degrees
            if avg_angle < 162:
                self.down_frames += 1
                if self.down_frames >= self.MIN_DOWN_FRAMES:
                    self.stage = "down"
                    self.up_frames = 0
            else:
                self.down_frames = 0

        elif self.stage == "down":
            debug_msg = "KALK!"
            debug_color = (255, 0, 255)
            
            # Back to standing when angle goes above 165
            if avg_angle > 165:
                self.up_frames += 1
                if self.up_frames >= self.MIN_UP_FRAMES:
                    self.counter += 1
                    self.stage = "up"
                    self.down_frames = 0
                    self.ref_hip_y = current_hip_y
                    if self.counter >= TARGET_REPS:
                        self.finished = True
            else:
                self.up_frames = 0

        return avg_angle, debug_msg, debug_color

    def reset(self):
        self.counter = 0
        self.stage = "calibrating"
        self.finished = False
        self.angle_history.clear()
        self.ref_hip_y = 0
        self.ref_leg_len = 0
        self.calibration_frames = 0
        self.down_frames = 0
        self.up_frames = 0


def _get_numpy_from_buffer_rgb(buffer, width, height):
    """Extract numpy array from GStreamer buffer (RGB format)"""
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        return None
    try:
        expected_size = width * height * 3
        if map_info.size >= expected_size:
            arr = np.frombuffer(map_info.data[:expected_size], dtype=np.uint8)
            return arr.reshape((height, width, 3)).copy()
    finally:
        buffer.unmap(map_info)
    return None


def _pad_probe_callback(pad, info):
    """Pad probe callback - processes buffer with Hailo metadata"""
    global _latest_frame_data, _squat_logic, _debug_counter
    
    # Debug counter for periodic logging
    if not hasattr(_pad_probe_callback, 'debug_counter'):
        _pad_probe_callback.debug_counter = 0
    _pad_probe_callback.debug_counter += 1
    
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    width, height = FRAME_W, FRAME_H

    # Get frame as numpy array (RGB format)
    frame_rgb = _get_numpy_from_buffer_rgb(buffer, width, height)
    if frame_rgb is None:
        return Gst.PadProbeReturn.OK

    # Convert RGB to BGR for OpenCV
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Get Hailo detections
    detections = []
    try:
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    except Exception as e:
        if _pad_probe_callback.debug_counter % 100 == 1:
            print(f"DEBUG: ROI error: {e}")

    # Debug: log detection count periodically
    if _pad_probe_callback.debug_counter % 60 == 1:
        print(f"DEBUG: Frame {_pad_probe_callback.debug_counter}, Detections: {len(detections)}")

    # Find largest person
    person = None
    max_area = 0.0
    for d in detections:
        try:
            if d.get_label() == "person":
                bbox = d.get_bbox()
                area = bbox.width() * bbox.height()
                if area > max_area:
                    max_area = area
                    person = d
        except Exception:
            continue

    current_angle = 0
    msg = "DIK DUR"
    col = (255, 255, 255)

    if person and _squat_logic:
        try:
            bbox = person.get_bbox()
            xmin = int(bbox.xmin() * width)
            ymin = int(bbox.ymin() * height)
            xmax = int(bbox.xmax() * width)
            ymax = int(bbox.ymax() * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            landmarks = person.get_objects_typed(hailo.HAILO_LANDMARKS)
            
            # Debug: log landmark info
            if _pad_probe_callback.debug_counter % 60 == 1:
                print(f"DEBUG: Person found, Landmarks count: {len(landmarks)}")
                if len(landmarks) > 0:
                    pts = landmarks[0].get_points()
                    print(f"DEBUG: Points count: {len(pts)}")
                    if len(pts) >= 17:
                        # Debug raw point values
                        p0 = pts[0]  # nose
                        p11 = pts[11]  # left hip
                        print(f"DEBUG: Raw point 0 (nose): x={p0.x()}, y={p0.y()}")
                        print(f"DEBUG: Raw point 11 (l_hip): x={p11.x()}, y={p11.y()}")
                        print(f"DEBUG: BBox: xmin={bbox.xmin()}, ymin={bbox.ymin()}, w={bbox.width()}, h={bbox.height()}")
                        # Using official Hailo formula: (point.x() * bbox.width() + bbox.xmin()) * width
                        px0 = int((p0.x() * bbox.width() + bbox.xmin()) * width)
                        py0 = int((p0.y() * bbox.height() + bbox.ymin()) * height)
                        px11 = int((p11.x() * bbox.width() + bbox.xmin()) * width)
                        py11 = int((p11.y() * bbox.height() + bbox.ymin()) * height)
                        print(f"DEBUG: Using official formula - Nose: ({px0}, {py0}), Hip: ({px11}, {py11})")
                        print(f"DEBUG: BBox: ({xmin}, {ymin}) to ({xmax}, {ymax})")
            
            if len(landmarks) > 0:
                points = landmarks[0].get_points()
                num_points = len(points)
                
                if num_points >= 17:
                    # Get bbox parameters once
                    bbox_x = bbox.xmin()
                    bbox_y = bbox.ymin()
                    bbox_w = bbox.width()
                    bbox_h = bbox.height()
                    
                    def get_xy(idx):
                        """Get pixel coordinates using exact Hailo example formula:
                           x = (point.x() * bbox.width() + bbox.xmin()) * width
                           y = (point.y() * bbox.height() + bbox.ymin()) * height
                        """
                        p = points[idx]
                        # point.x()/y() should be 0-1 normalized relative to bbox
                        px = int((p.x() * bbox_w + bbox_x) * width)
                        py = int((p.y() * bbox_h + bbox_y) * height)
                        
                        # Clamp to valid range
                        px = max(0, min(width - 1, px))
                        py = max(0, min(height - 1, py))
                        return (px, py)

                    # Get all keypoints
                    nose = get_xy(0)
                    l_shoulder = get_xy(5)
                    r_shoulder = get_xy(6)
                    l_elbow = get_xy(7)
                    r_elbow = get_xy(8)
                    l_wrist = get_xy(9)
                    r_wrist = get_xy(10)
                    l_hip = get_xy(11)
                    r_hip = get_xy(12)
                    l_knee = get_xy(13)
                    r_knee = get_xy(14)
                    l_ankle = get_xy(15)
                    r_ankle = get_xy(16)

                    # Use both legs for squat detection
                    current_angle, msg, col = _squat_logic.process(
                        l_hip, l_knee, l_ankle,
                        r_hip, r_knee, r_ankle
                    )
                    
                    # Debug coordinates
                    if _pad_probe_callback.debug_counter % 60 == 1:
                        print(f"DEBUG: L_knee:{l_knee}, R_knee:{r_knee}")
                        print(f"DEBUG: Angle:{current_angle}, Stage:{_squat_logic.stage}, Counter:{_squat_logic.counter}")

                    # Draw skeleton
                    skeleton_color = (0, 255, 255)
                    joint_color = (0, 0, 255)
                    line_thickness = 3
                    joint_radius = 5
                    
                    cv2.circle(frame, nose, joint_radius, (255, 0, 255), -1)
                    cv2.line(frame, l_shoulder, r_shoulder, skeleton_color, line_thickness)
                    cv2.line(frame, l_shoulder, l_hip, skeleton_color, line_thickness)
                    cv2.line(frame, r_shoulder, r_hip, skeleton_color, line_thickness)
                    cv2.line(frame, l_hip, r_hip, skeleton_color, line_thickness)
                    cv2.line(frame, l_shoulder, l_elbow, (255, 165, 0), line_thickness)
                    cv2.line(frame, l_elbow, l_wrist, (255, 165, 0), line_thickness)
                    cv2.line(frame, r_shoulder, r_elbow, (255, 165, 0), line_thickness)
                    cv2.line(frame, r_elbow, r_wrist, (255, 165, 0), line_thickness)
                    cv2.line(frame, l_hip, l_knee, (0, 255, 0), line_thickness + 2)
                    cv2.line(frame, l_knee, l_ankle, (0, 255, 0), line_thickness + 2)
                    cv2.line(frame, r_hip, r_knee, skeleton_color, line_thickness)
                    cv2.line(frame, r_knee, r_ankle, skeleton_color, line_thickness)
                    
                    for joint in [l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip]:
                        cv2.circle(frame, joint, joint_radius, joint_color, -1)
                    
                    cv2.circle(frame, l_hip, joint_radius + 3, (0, 255, 0), -1)
                    cv2.circle(frame, l_knee, joint_radius + 3, (0, 255, 0), -1)
                    cv2.circle(frame, l_ankle, joint_radius + 3, (0, 255, 0), -1)
                    cv2.circle(frame, r_knee, joint_radius, joint_color, -1)
                    cv2.circle(frame, r_ankle, joint_radius, joint_color, -1)
                    
                    cv2.putText(frame, f"{current_angle}", (l_knee[0] + 15, l_knee[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        except Exception as e:
            if _pad_probe_callback.debug_counter % 60 == 1:
                print(f"DEBUG: Drawing error: {e}")

    # Draw overlay
    counter = _squat_logic.counter if _squat_logic else 0
    finished = _squat_logic.finished if _squat_logic else False
    
    cv2.rectangle(frame, (0, 0), (width, 80), (30, 30, 30), -1)
    cv2.putText(frame, f"REPS: {counter}/{TARGET_REPS}", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(frame, msg, (width - 250, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)

    # Convert to RGBA for UI
    frame_rgba_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    with _lock:
        _latest_frame_data = (frame_rgba_out.tobytes(), width, height, counter, finished)

    return Gst.PadProbeReturn.OK


def _on_bus_message(bus, message, loop):
    """Handle GStreamer bus messages"""
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End of stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"GStreamer Error: {err}, {debug}")
        loop.quit()
    return True


def _build_pipeline_string():
    """Build GStreamer pipeline with appsrc for Picamera2 input"""
    pipeline = (
        # Appsrc for Picamera2 frames
        f"appsrc name=camera_source is-live=true format=time ! "
        f"video/x-raw,format=RGB,width={FRAME_W},height={FRAME_H},framerate=30/1 ! "
        f"queue leaky=downstream max-size-buffers=2 ! "
        
        # Hailo inference (force-writable to handle appsrc buffers)
        f"hailonet hef-path={MY_HEF_PATH} batch-size=1 force-writable=true ! "
        f"queue leaky=downstream max-size-buffers=3 ! "
        f"hailofilter so-path={MY_POST_SO} function-name={MY_POST_FUNC} qos=false ! "
        f"queue leaky=downstream max-size-buffers=3 ! "
        
        # Identity for pad probe
        f"identity name=identity_callback ! "
        f"queue leaky=downstream max-size-buffers=2 ! "
        
        # Fakesink
        f"fakesink sync=false"
    )
    return pipeline


def _camera_feed_thread():
    """Thread that captures from Picamera2 and pushes to GStreamer appsrc"""
    global _picam2, _pipeline, _stop_flag
    
    appsrc = _pipeline.get_by_name("camera_source")
    if appsrc is None:
        print("Error: Could not find appsrc")
        return

    frame_duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)
    frame_count = 0
    
    print("Camera feed thread started")
    
    while not _stop_flag:
        try:
            # Capture frame from Picamera2
            frame = _picam2.capture_array("main")
            if frame is None:
                continue
            
            # Resize to model input size
            frame_resized = cv2.resize(frame, (FRAME_W, FRAME_H))
            
            # Convert BGR to RGB if needed (Picamera2 outputs RGB888)
            if len(frame_resized.shape) == 3 and frame_resized.shape[2] == 3:
                # Already RGB from Picamera2 with RGB888 format
                frame_rgb = frame_resized
            else:
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Create GStreamer buffer
            data = frame_rgb.tobytes()
            buffer = Gst.Buffer.new_wrapped(data)
            buffer.pts = frame_count * frame_duration
            buffer.duration = frame_duration
            
            # Push to appsrc
            ret = appsrc.emit("push-buffer", buffer)
            if ret != Gst.FlowReturn.OK:
                if ret == Gst.FlowReturn.FLUSHING:
                    break
                print(f"Error pushing buffer: {ret}")
                break
            
            frame_count += 1
            
        except Exception as e:
            print(f"Camera capture error: {e}")
            time.sleep(0.1)
    
    # Send EOS
    appsrc.emit("end-of-stream")
    print("Camera feed thread ended")


def _run_main_loop():
    """Run the GLib main loop in a separate thread"""
    global _main_loop
    try:
        _main_loop.run()
    except Exception as e:
        print(f"Main loop error: {e}")
    finally:
        print("GLib main loop ended")


def start_camera():
    """Start the camera and Hailo pipeline (non-blocking)"""
    global _pipeline, _main_loop, _loop_thread, _camera_thread, _picam2
    global _is_running, _stop_flag, _squat_logic

    with _lock:
        if _is_running:
            print("Camera already running")
            return True
        _is_running = True
        _stop_flag = False

    print("Starting Hailo camera pipeline...")

    try:
        # Initialize squat logic
        _squat_logic = SquatLogic()

        # Initialize Picamera2
        if Picamera2 is None:
            raise Exception("Picamera2 not available")
        
        _picam2 = Picamera2()
        config = _picam2.create_video_configuration(
            main={"size": (CAMERA_W, CAMERA_H), "format": "RGB888"},
            controls={"FrameRate": 30}
        )
        _picam2.configure(config)
        _picam2.start()
        print("Picamera2 started")
        time.sleep(0.5)  # Let camera warm up

        # Build GStreamer pipeline
        pipeline_str = _build_pipeline_string()
        print(f"Pipeline: {pipeline_str[:80]}...")
        
        _pipeline = Gst.parse_launch(pipeline_str)
        if _pipeline is None:
            raise Exception("Failed to create pipeline")

        # Add pad probe to identity element
        identity = _pipeline.get_by_name("identity_callback")
        if identity is None:
            raise Exception("Could not find identity_callback element")
        identity_pad = identity.get_static_pad("src")
        identity_pad.add_probe(Gst.PadProbeType.BUFFER, _pad_probe_callback)

        # Create main loop
        _main_loop = GLib.MainLoop()

        # Setup bus
        bus = _pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", _on_bus_message, _main_loop)

        # Start pipeline
        ret = _pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise Exception("Failed to start pipeline")

        print("Pipeline started")

        # Start camera feed thread
        _camera_thread = threading.Thread(target=_camera_feed_thread, daemon=True)
        _camera_thread.start()

        # Run main loop in separate thread
        _loop_thread = threading.Thread(target=_run_main_loop, daemon=True)
        _loop_thread.start()

        time.sleep(0.5)
        print("Hailo camera ready!")
        return True

    except Exception as e:
        print(f"Failed to start camera: {e}")
        import traceback
        traceback.print_exc()
        with _lock:
            _is_running = False
        stop_camera()
        return False


def stop_camera():
    """Stop the camera and cleanup resources"""
    global _pipeline, _main_loop, _loop_thread, _camera_thread, _picam2
    global _is_running, _stop_flag, _latest_frame_data

    print("Stopping camera...")

    _stop_flag = True

    with _lock:
        _is_running = False
        _latest_frame_data = None

    # Stop camera thread
    if _camera_thread and _camera_thread.is_alive():
        _camera_thread.join(timeout=2.0)
    _camera_thread = None

    # Stop main loop
    if _main_loop and _main_loop.is_running():
        _main_loop.quit()

    # Stop pipeline
    if _pipeline:
        _pipeline.set_state(Gst.State.NULL)
        _pipeline = None

    # Stop Picamera2
    if _picam2:
        try:
            _picam2.stop()
            _picam2.close()
        except Exception:
            pass
        _picam2 = None

    # Wait for loop thread
    if _loop_thread and _loop_thread.is_alive():
        _loop_thread.join(timeout=2.0)
    _loop_thread = None
    _main_loop = None

    print("Camera stopped")


def get_ai_frame():
    """Get the latest processed frame (non-blocking)"""
    global _latest_frame_data

    with _lock:
        if _is_running:
            return _latest_frame_data

    # Start camera if not running
    if not _is_running:
        if not start_camera():
            return None
        # Wait for first frame
        for _ in range(50):
            time.sleep(0.1)
            with _lock:
                if _latest_frame_data is not None:
                    return _latest_frame_data
        return None

    with _lock:
        return _latest_frame_data


def reset_exercise():
    """Reset the squat counter and state"""
    global _squat_logic
    if _squat_logic:
        _squat_logic.reset()
        print("Exercise reset")


def is_running():
    """Check if camera is running"""
    with _lock:
        return _is_running


# Cleanup on module unload
import atexit
atexit.register(stop_camera)
