import requests

class OllamaService:
    def __init__(self, model="gemma2"):
        self.base_url = "http://localhost:11434/api/generate"
        self.model = model

    def generate_response(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API 호출 중 오류: {e}")