from flask import Flask, request, jsonify
import base64, tempfile, os, numpy as np, cv2
from face_analysis import analyze_face, _process_frame
import mediapipe as mp
from speech_analysis import analyze_speech

app = Flask(__name__)
mp_face_mesh = mp.solutions.face_mesh

# ========== FACE ENDPOINTS ==========

@app.route("/analyze/face", methods=["POST"])
def face_from_image():
    """
    รับรูปภาพ base64 → วิเคราะห์ใบหน้า → คืนผล
    FlutterFlow ส่ง: { "image": "<base64 string>" }
    """
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "ต้องส่ง image เป็น base64"}), 400

    # แปลง base64 เป็นไฟล์รูป
    try:
        img_bytes = base64.b64decode(data["image"])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "แปลงรูปไม่ได้"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # วิเคราะห์
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        result = _process_frame(frame, face_mesh)

    return jsonify(result)


# ========== SPEECH ENDPOINTS ==========

@app.route("/analyze/speech", methods=["POST"])
def speech_from_audio():
    """
    รับไฟล์เสียง base64 → วิเคราะห์การพูด → คืนผล
    FlutterFlow ส่ง: { "audio": "<base64>", "target": "ประโยคเป้าหมาย" }
    """
    data = request.get_json()
    if not data or "audio" not in data:
        return jsonify({"error": "ต้องส่ง audio เป็น base64"}), 400

    # บันทึกไฟล์เสียงชั่วคราว
    try:
        audio_bytes = base64.b64decode(data["audio"])
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # วิเคราะห์
    target = data.get("target", None)
    result = analyze_speech(target_sentence=target, audio_path=tmp.name)

    # ลบไฟล์ temp
    if os.path.exists(tmp.name):
        os.remove(tmp.name)

    return jsonify(result)


# ========== REHAB ENDPOINTS ==========

@app.route("/rehab/daily-score", methods=["POST"])
def save_daily_score():
    """
    บันทึกคะแนนฝึกประจำวัน
    FlutterFlow ส่ง: { "user_id": "xxx", "face_score": 80, "speech_score": 75, "date": "2026-04-04" }
    """
    data = request.get_json()
    required = ["user_id", "face_score", "speech_score", "date"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"ต้องมี field: {field}"}), 400

    # ในแอปจริง บันทึกลง Firebase/Google Sheets ที่นี่
    # ตัวอย่างนี้แค่ return สรุปกลับ
    total = round((data["face_score"] * 0.5 + data["speech_score"] * 0.5), 1)

    if total >= 80:
        summary = "ยอดเยี่ยม! พัฒนาการดีมากครับ"
        badge = "gold"
    elif total >= 60:
        summary = "ดีขึ้น! ฝึกต่อเนื่องทุกวันนะครับ"
        badge = "silver"
    else:
        summary = "ค่อยๆ ฝึกครับ ไม่ต้องรีบ"
        badge = "bronze"

    return jsonify({
        "user_id":      data["user_id"],
        "date":         data["date"],
        "face_score":   data["face_score"],
        "speech_score": data["speech_score"],
        "total_score":  total,
        "summary":      summary,
        "badge":        badge,
    })


@app.route("/rehab/exercises", methods=["GET"])
def get_exercises():
    """
    คืนรายการแบบฝึกหัดตามคะแนนล่าสุด
    """
    score = request.args.get("score", 50, type=float)

    if score >= 80:
        exercises = [
            {"id": 1, "name": "ฝึกยิ้มกว้าง", "duration": "5 วินาที x 15 ครั้ง", "level": "กลาง"},
            {"id": 2, "name": "พูดประโยคยาว", "duration": "3 ประโยค x 3 รอบ", "level": "กลาง"},
            {"id": 3, "name": "ฝึกออกเสียงพยัญชนะ", "duration": "10 นาที", "level": "กลาง"},
        ]
    elif score >= 50:
        exercises = [
            {"id": 1, "name": "ยิ้มค้าง", "duration": "5 วินาที x 10 ครั้ง", "level": "ง่าย"},
            {"id": 2, "name": "พูดคำสั้น", "duration": "5 คำ x 3 รอบ", "level": "ง่าย"},
            {"id": 3, "name": "เป่าลม", "duration": "3 วินาที x 10 ครั้ง", "level": "ง่าย"},
        ]
    else:
        exercises = [
            {"id": 1, "name": "ขยับปากช้าๆ", "duration": "2 นาที", "level": "เริ่มต้น"},
            {"id": 2, "name": "พูดสระ อา อี อู", "duration": "5 นาที", "level": "เริ่มต้น"},
        ]

    return jsonify({"score": score, "exercises": exercises})


# ========== HEALTH CHECK ==========

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "app": "StrokeGuard AI API",
        "version": "1.0",
        "endpoints": [
            "POST /analyze/face",
            "POST /analyze/speech",
            "POST /rehab/daily-score",
            "GET  /rehab/exercises?score=75",
        ]
    })


if __name__ == "__main__":
    print("🚀 StrokeGuard API กำลังรัน...")
    print("📡 เปิดที่ http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)