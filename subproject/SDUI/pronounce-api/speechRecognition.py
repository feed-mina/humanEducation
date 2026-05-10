# 필요한 도구 가져오기
import speech_recognition as sr   # 소리를 글자로 바꾸기
import Levenshtein               # 글자 차이 계산하기

# ① 원래 문장 준비
원래문장 = "안녕하세요 만나서 반갑습니다"

# ② 마이크로 말하기
recognizer = sr.Recognizer()
with sr.Microphone() as source:
    print("아래 문장을 읽어주세요!")
    print(f"읽을 문장: {원래문장}")
    print("녹음 시작합니다... 말해주세요!")

    # ③ 소리를 듣고 저장
    녹음 = recognizer.listen(source)

# ④ 소리를 글자로 바꾸기
try:
    발음문장 = recognizer.recognize_google(녹음, language="ko-KR")
    print(f"\n당신이 읽은 문장: {발음문장}")

    # ⑤ 원래 문장과 비교해서 글자 차이 계산
    거리 = Levenshtein.distance(원래문장, 발음문장)
    print(f"\n틀린 글자 수: {거리}")

    # ⑥ 발음 점수 매기기
    총글자수 = len(원래문장)
    점수 = ((총글자수 - 거리) / 총글자수) * 100
    print(f"발음 점수: {점수:.1f}점")

    # 결과 알려주기
    if 거리 == 0:
        print("완벽해요! 발음이 아주 좋아요.")
    elif 점수 >= 80:
        print("좋아요! 조금만 더 연습해요.")
    else:
        print("연습이 더 필요해요!")

except:
    print("죄송해요, 소리를 인식하지 못했어요.")
# 원래 문장과 발음 문장
원래문장 = "안녕하세요"
발음문장 = "안녀하세요"

# 글자별 비교
틀린글자 = []
for i in range(min(len(원래문장), len(발음문장))):
    if 원래문장[i] != 발음문장[i]:
        틀린글자.append(f"{i+1}번째 글자 틀림: '{원래문장[i]}' → '{발음문장[i]}'")

# 남은 글자가 더 길면 추가로 체크
if len(원래문장) > len(발음문장):
    for i in range(len(발음문장), len(원래문장)):
        틀린글자.append(f"{i+1}번째 글자 없음 (원래는 '{원래문장[i]}'이 있어야 함)")
elif len(발음문장) > len(원래문장):
    for i in range(len(원래문장), len(발음문장)):
        틀린글자.append(f"{i+1}번째 글자 추가됨: '{발음문장[i]}'")

# 결과 출력
if 틀린글자:
    print("틀린 부분 발견!")
    for t in 틀린글자:
        print(t)
else:
    print("완벽해요! 모든 글자가 맞아요.")
