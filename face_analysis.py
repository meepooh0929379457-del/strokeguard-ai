import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# จุดสำคัญบนใบหน้าที่ใช้วัดความสมมาตร
LANDMARKS = {
    "left_eye":   [33, 160, 158, 133, 153, 144],
    "right_eye":  [362, 385, 387, 263, 373, 380],
    "left_mouth": [61, 146, 91, 181, 84],
    "right_mouth":[291, 375, 321, 405, 314],
    "left_cheek": [234, 93, 132],
    "right_cheek":[454, 323, 361],
    "nose_tip":   [1],
    "chin":       [152],
}

def get_landmark_coords(landmarks, indices, w, h):
    return np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])

def calc_symmetry_score(left_pts, right_pts, center_x):
    # วัดระยะห่างซ้าย-ขวาจากแกนกลาง
    left_dist  = np.mean(np.abs(left_pts[:, 0]  - center_x))
    right_dist = np.mean(np.abs(right_pts[:, 0] - center_x))
    if max(left_dist, right_dist) == 0:
        return 100.0
    diff = abs(left_dist - right_dist)
    score = max(0, 100 - (diff / max(left_dist, right_dist)) * 100)
    return round(score, 1)

def analyze_face(image_path=None, use_camera=False):
    """
    วิเคราะห์ใบหน้าจากไฟล์รูปหรือกล้อง
    คืนค่า: dict ผลการวิเคราะห์
    """
    results_data = {}

    with mp_face_mesh.FaceMesh(
        static_image_mode=not use_camera,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        if use_camera:
            cap = cv2.VideoCapture(0)
            print("กำลังเปิดกล้อง... กด 'q' เพื่อหยุด")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results_data = _process_frame(frame, face_mesh)
                _draw_overlay(frame, results_data)
                cv2.imshow("StrokeGuard - Face Analysis", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

        else:
            frame = cv2.imread(image_path)
            if frame is None:
                return {"error": "ไม่พบไฟล์รูปภาพ"}
            results_data = _process_frame(frame, face_mesh)

    return results_data

def _process_frame(frame, face_mesh):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return {"error": "ตรวจไม่พบใบหน้า กรุณาหันหน้าตรงๆ"}

    lm = results.multi_face_landmarks[0].landmark

    # หาแกนกลางใบหน้า
    nose_x = lm[1].x * w

    # คำนวณคะแนนสมมาตรแต่ละส่วน
    eye_score    = calc_symmetry_score(
        get_landmark_coords(lm, LANDMARKS["left_eye"], w, h),
        get_landmark_coords(lm, LANDMARKS["right_eye"], w, h),
        nose_x
    )
    mouth_score  = calc_symmetry_score(
        get_landmark_coords(lm, LANDMARKS["left_mouth"], w, h),
        get_landmark_coords(lm, LANDMARKS["right_mouth"], w, h),
        nose_x
    )
    cheek_score  = calc_symmetry_score(
        get_landmark_coords(lm, LANDMARKS["left_cheek"], w, h),
        get_landmark_coords(lm, LANDMARKS["right_cheek"], w, h),
        nose_x
    )

    overall = round((eye_score * 0.3 + mouth_score * 0.5 + cheek_score * 0.2), 1)

    # ประเมินผล
    if overall >= 85:
        status  = "ปกติ"
        advice  = "ใบหน้าสมมาตรดี ทำแบบฝึกหัดบำรุงรักษาต่อไปครับ"
        level   = "good"
    elif overall >= 65:
        status  = "เบี้ยวเล็กน้อย"
        advice  = "แนะนำฝึกยิ้มกว้างๆ ค้างไว้ 5 วินาที x 10 ครั้ง/วัน"
        level   = "warning"
    else:
        status  = "ควรพบแพทย์"
        advice  = "ความสมมาตรต่ำมาก กรุณาปรึกษาแพทย์ก่อนฝึกเองครับ"
        level   = "danger"

    return {
        "overall_score": overall,
        "eye_symmetry":   eye_score,
        "mouth_symmetry": mouth_score,
        "cheek_symmetry": cheek_score,
        "status":  status,
        "advice":  advice,
        "level":   level,
    }

def _draw_overlay(frame, data):
    if "error" in data:
        cv2.putText(frame, data["error"], (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return

    h, w = frame.shape[:2]
    color = {"good": (0,200,0), "warning": (0,165,255), "danger": (0,0,255)}
    c = color.get(data["level"], (255,255,255))

    cv2.putText(frame, f"คะแนนรวม: {data['overall_score']}%", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
    cv2.putText(frame, f"ดวงตา: {data['eye_symmetry']}%", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f"ปาก:   {data['mouth_symmetry']}%", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f"แก้ม:  {data['cheek_symmetry']}%", (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, data["status"], (w-180, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)

# ทดสอบรันตรงๆ
if __name__ == "__main__":
    print("เปิดกล้องวิเคราะห์ใบหน้า...")
    result = analyze_face(use_camera=True)
    print("\n=== ผลการวิเคราะห์ ===")
    for k, v in result.items():
        print(f"  {k}: {v}")