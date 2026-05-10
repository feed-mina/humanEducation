import requests
key = "435286d5f656c30b88ef6975996eea8e"
query = "수원 행궁동 몽테드"
url = f"https://dapi.kakao.com/v2/local/search/keyword.json?query={query}"
headers = {"Authorization": f"KakaoAK {key}"}
res = requests.get(url, headers=headers)
print(res.json())
