import whisper
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile, os

# โหลด Whisper model (ครั้งแรกจะโหลดนานหน่อย)
print("กำลังโหลด Whisper model...")
_whisper_model = whisper.load_model("base")

# ประโยคฝึกพูดสำหรับผู้ป่วย Stroke
PRACTICE_SENTENCES = [
    "กินข้าวแล้วหรือยัง",
    "สวัสดีครับ ผมสบายดี",
    "วันนี้อากาศดีมาก",
    "ขอบคุณมากครับ",
    "ฉันต้องการน้ำ",
]

def record_audio(duration=5, sample_rate=16000):
    """
    อัดเสียงจากไมค์ duration วินาที
    """
    print(f"🎙️ กำลังอัดเสียง {duration} วินาที... พูดได้เลย!")
    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate,
                   channels=1, dtype='float32')
    sd.wait()
    print("✅ อัดเสียงเสร็จแล้ว")
    return audio.flatten(), sample_rate

def save_temp_audio(audio, sample_rate):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sample_rate)
    return tmp.name

def analyze_speech_features(audio, sample_rate):
    """
    วิเคราะห์คุณภาพเสียงด้วย librosa
    """
    # ความดังเฉลี่ย (ตรวจว่าพูดออกมาไหม)
    rms = float(np.sqrt(np.mean(audio**2)))

    # ความเร็วพูด — นับจาก onset events
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sample_rate)
    speech_rate = len(onset_frames) / (len(audio) / sample_rate)

    # ความชัดเจนของเสียง — ผ่าน spectral clarity
    stft = np.abs(librosa.stft(audio))
    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(S=stft)))
    clarity = max(0, min(100, (1 - spectral_flatness * 10) * 100))

    # MFCC — ลักษณะเสียงพูด
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1).tolist()

    return {
        "rms_volume":   round(rms * 1000, 2),
        "speech_rate":  round(speech_rate, 2),
        "clarity":      round(clarity, 1),
        "mfcc_mean":    mfcc_mean,
    }

def analyze_speech(target_sentence=None, audio_path=None, record=False, duration=5):
    """
    วิเคราะห์การพูดแบบครบวงจร
    - target_sentence: ประโยคที่ควรพูด (ถ้ามี)
    - audio_path: ไฟล์เสียงที่มีอยู่แล้ว
    - record: อัดเสียงจากไมค์เลย
    """
    # 1. เตรียมไฟล์เสียง
    if record:
        audio, sr = record_audio(duration=duration)
        audio_path = save_temp_audio(audio, sr)
    elif audio_path:
        audio, sr = librosa.load(audio_path, sr=16000)
    else:
        return {"error": "ไม่มีแหล่งเสียง กรุณาระบุ audio_path หรือ record=True"}

    # 2. แปลงเสียงเป็นข้อความด้วย Whisper
    print("🔍 กำลังวิเคราะห์เสียง...")
    whisper_result = _whisper_model.transcribe(audio_path, language="th")
    transcribed = whisper_result["text"].strip()

    # 3. วิเคราะห์คุณสมบัติเสียง
    if audio_path and not record:
        audio, sr = librosa.load(audio_path, sr=16000)
    features = analyze_speech_features(audio, sr)

    # 4. เปรียบเทียบกับประโยคเป้าหมาย
    match_score = 0
    if target_sentence:
        # คำนวณความตรงกันแบบง่าย
        target_words = set(target_sentence.replace(" ", ""))
        spoken_words = set(transcribed.replace(" ", ""))
        if len(target_words) > 0:
            intersection = len(target_words & spoken_words)
            match_score = round((intersection / len(target_words)) * 100, 1)

    # 5. ประเมินผลรวม
    volume_ok = features["rms_volume"] > 5
    clarity_ok = features["clarity"] > 50
    has_speech = len(transcribed) > 0

    if not has_speech or not volume_ok:
        status = "ไม่ได้ยินเสียง"
        advice = "พูดให้ดังขึ้น และถือโทรศัพท์ใกล้ปากมากขึ้นครับ"
        level  = "danger"
        score  = 0
    elif not clarity_ok:
        status = "เสียงไม่ชัด"
        advice = "ลองพูดช้าๆ ออกเสียงให้ชัดทีละพยางค์ครับ"
        level  = "warning"
        score  = 40
    elif match_score >= 70 or not target_sentence:
        status = "พูดได้ดี"
        advice = "เยี่ยมมาก! ลองฝึกประโยคต่อไปได้เลยครับ"
        level  = "good"
        score  = min(100, round((features["clarity"] + match_score) / 2))
    else:
        status = "ต้องฝึกเพิ่ม"
        advice = f"พูดได้ {match_score}% ลองพูดอีกครั้งช้าๆ ครับ"
        level  = "warning"
        score  = match_score

    # ลบไฟล์ temp
    if record and os.path.exists(audio_path):
        os.remove(audio_path)

    return {
        "transcribed_text": transcribed,
        "target_sentence":  target_sentence or "-",
        "match_score":      match_score,
        "overall_score":    score,
        "clarity":          features["clarity"],
        "volume":           features["rms_volume"],
        "speech_rate":      features["speech_rate"],
        "status":           status,
        "advice":           advice,
        "level":            level,
    }

def daily_practice_session():
    """
    โหมดฝึกพูดประจำวัน — ทดสอบทีละประโยค
    """
    print("\n=== โหมดฝึกพูดประจำวัน ===")
    scores = []

    for i, sentence in enumerate(PRACTICE_SENTENCES, 1):
        print(f"\nประโยคที่ {i}/{len(PRACTICE_SENTENCES)}: '{sentence}'")
        input("กด Enter เมื่อพร้อมพูด...")

        result = analyze_speech(
            target_sentence=sentence,
            record=True,
            duration=6
        )

        print(f"  ได้ยิน: '{result['transcribed_text']}'")
        print(f"  คะแนน: {result['overall_score']}% — {result['status']}")
        print(f"  คำแนะนำ: {result['advice']}")
        scores.append(result["overall_score"])

    avg = round(sum(scores) / len(scores), 1)
    print(f"\n=== ผลรวมวันนี้: {avg}% ===")
    if avg >= 80:
        print("🎉 ยอดเยี่ยม! ฝึกต่อเนื่องทุกวันนะครับ")
    elif avg >= 60:
        print("💪 ดีขึ้น! พรุ่งนี้ลองอีกครั้งครับ")
    else:
        print("🙏 ไม่เป็นไร ค่อยๆ ฝึกทุกวันครับ")

    return {"average_score": avg, "detail_scores": scores}

# ทดสอบ
if __name__ == "__main__":
    daily_practice_session()