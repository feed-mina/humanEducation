from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_translate_only():
    response = client.post("/translate_only", json={"content": "오늘 날씨 어때?"})
    assert response.status_code == 200
    assert "translated_text" in response.json()

def test_tts_blob():
    response = client.post("/tts_blob", json={"text": "これはテストです。"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
