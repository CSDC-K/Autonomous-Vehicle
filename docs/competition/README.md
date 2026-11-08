# 18th International MEB Robot Competition — Autonomous Vehicle Category (2026)

The 18th International MEB Robot Competition was organised by the Turkish Ministry of National
Education (MEB) in **Antalya, 6–8 May 2026**. This folder keeps the official category guide that
this car was built for.

| File | Language | Source |
|---|---|---|
| [`MEB_2026_Autonomous_Vehicle_Category_Guide_EN.pdf`](MEB_2026_Autonomous_Vehicle_Category_Guide_EN.pdf) | English (official MEB translation) | [robot.meb.gov.tr](https://robot.meb.gov.tr/yukleme/Robot2026/10_otonom_arac_ENG.pdf) |
| [`MEB_2026_Otonom_Arac_Kategori_Kilavuzu_TR.pdf`](MEB_2026_Otonom_Arac_Kategori_Kilavuzu_TR.pdf) | Turkish (original) | [robot.meb.gov.tr](https://robot.meb.gov.tr/yukleme/Robot2026/10_otonom_arac.pdf) |

Both documents are © T.C. Milli Eğitim Bakanlığı and are included here unchanged for reference.
The latest versions and announcements are always on [robot.meb.gov.tr](https://robot.meb.gov.tr).

---

## Rules summary

A short English summary of the parts of the guide that shaped this project. The PDF is the
authoritative version.

### Vehicle limits
- Must fit in a **20 × 30 cm** box and be at most **25 cm** tall; wheels at most 10 cm in diameter. No weight limit.
- **Camera only.** Distance sensors, lidar, IR, ultrasonic or any other sensor → disqualification. Any number of cameras is allowed.
- Any controller board and any motors are allowed.
- The control software must be **written entirely by the team** (no ready-made, commercial or block-based control software).
- Wi-Fi, Bluetooth, IR and RF must be **off** during a run; remote control → disqualification.

### Track
- Black surface with white solid and dashed lane markings; the car must stay in its own lane.
- A traffic light at the start, task areas, overtaking zones, a dead-end section and coloured parking areas at the end.
- An **orange vehicle (20 × 30 × 25 cm)** to overtake, and **yellow vehicles (20 × 45 × 25 cm)** as obstacles in no-overtaking areas.
- The referees can change the order and position of pedestrian crossings, level crossings and speed bumps for every competitor.

### Tasks and points

| # | Task | What the car must do | Points |
|---|---|---|---|
| 1 | Traffic light start | Start within 3 s of green, using pre-loaded software and a button (no external computer) | 50 (25 on 2nd try) |
| 2 | Pedestrian crossing | Stop ≤ 30 cm before the crossing and wait ≥ 5 s | 50 |
| 3 | Speed bump | Drive over it without leaving the lane | 50 |
| 4 | Level crossing | Stop ≤ 30 cm before it and wait ≥ 5 s | 50 |
| 5 | Overtaking | Overtake the orange vehicle, only inside the overtaking zone | 100 |
| 6 | Dead end | Recognise the dead end and turn right instead of entering it | 100 |
| 7 | Parking | Stop inside the **red** parking area (red/blue/green positions change every run) | 100 |
| 8 | Zone completion | Complete the task zones without leaving the track or crossing into the opposite lane | 50 |

- One lap has a maximum of **4 minutes (240 s)**.
- Cars that finish get a time bonus: **240 − finish time (s)**. Ties are broken by the faster time.
- If the car leaves the track or gets stuck, it can be put back with the referee's approval, but that task earns no points.

---

## How this project relates to the rules

- **Camera only, own software:** the car uses one Raspberry Pi camera and runs only the code in this repository.
- **Small chassis:** the 20 × 30 × 25 cm limit is why the car is small; the low base speed in the final `lane_opencv.py` comes from the small chassis oscillating at high speed.
- **Lane keeping and zone completion (task 8):** the final OpenCV lane pipeline (`lane_opencv.py`).
- **Pedestrian crossing (task 2):** `lane_opencv.py` detects crossing stripes, removes them from the lane mask and slows down.
- **Signs, lights and vehicles (tasks 1, 3–7):** the `road_signs` dataset and model cover the traffic lights (`kirmizi_isik`, `sari_isik`, `yesil_isik`), pedestrian crossing (`yaya_gecidi`), speed bump (`tumsek`), level crossing (`hemzemin`), overtaking allowed (`sollama_serbest`), dead end (`cikmaz_yol`), parking (`park`), and the orange and yellow vehicles (`turuncu_kutu`, `sari_kutu`). The model was tested separately with `road_sign_test.py`.
