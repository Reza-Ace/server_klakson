from flask import Flask, request, jsonify
import numpy as np
import soundfile as sf
import io
import tensorflow as tf
import librosa

app = Flask(__name__)

# Load model Tersimpan
model = tf.keras.models.load_model("horn_detection_model.h5")

def predict_from_raw_bytes(raw_bytes):
    # Ubah bytes mentah (ESP32) ke numpy array int16
    audio_np = np.frombuffer(raw_bytes, dtype=np.int16)

    # Ubah ke float32 range -1.0 ~ 1.0
    audio = audio_np.astype(np.float32) / 32768.0

    # Simpan ke buffer sebagai .wav agar bisa dibaca oleh librosa
    buffer = io.BytesIO()
    sf.write(buffer, audio, samplerate=16000, format='WAV')
    buffer.seek(0)

    # Baca ulang dengan librosa
    y, sr = librosa.load(buffer, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = mfcc.T  # Transpose agar (time_steps, n_mfcc)

    # Padding/trimming agar jadi ukuran (32, 40)
    if mfcc.shape[0] < 32:
        pad_width = 32 - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
    else:
        mfcc = mfcc[:32, :]

    # Tambahkan batch dimension: (1, 32, 40)
    mfcc_input = np.expand_dims(mfcc, axis=0)

    # Prediksi
    pred = model.predict(mfcc_input)[0][0]
    is_klakson = pred > 0.5

    return {"klakson": bool(is_klakson), "skor": float(pred)}

@app.route("/predict", methods=["POST"])
def predict():
    if request.data:
        result = predict_from_raw_bytes(request.data)
        return jsonify(result)
    else:
        return jsonify({"error": "No audio file received"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
