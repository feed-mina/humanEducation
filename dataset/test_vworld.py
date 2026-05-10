import requests
key = "635AACC4-9885-311C-A0C5-D4A64E2ED023"
query = "수원 행궁동 몽테드"
url = f"http://api.vworld.kr/req/search?service=search&request=search&type=place&format=json&query={query}&key={key}"
res = requests.get(url)
print(res.json())
