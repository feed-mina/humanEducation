from flask import Flask, request, render_template, jsonify
import speech_recognition as sr

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    file = request.files['audio']
    recognizer = sr.Recognizer()
    with sr.AudioFile(file) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="ko-KR")
            return jsonify({'text': text})
        except:
            return jsonify({'text': "음성 인식 실패"})

if __name__ == '__main__':
    app.run(debug=True)
