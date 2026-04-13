"""
로컬 PC (CPU 기반)에서 Gemma2 모델과 LoRA 어댑터를 로드하여 질문에 답하는 프로그램

필수 설치:
pip install transformers peft torch accelerate huggingface-hub

주의: CPU 환경에서는 모델 로딩 및 답변 생성 시간이 매우 오래 걸립니다. (수십분 ~ 수시간)
      8GB 이상의 RAM과 넉넉한 스왑 공간이 필수입니다.
"""

# 사전 설치 라이브러리 : pip install -U transformers peft torch accelerate huggingface-hub
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login
import os
import time # 시간 측정을 위한 모듈

# ============================================================
# 1. 설정 영역 (사용자가 수정해야 할 부분)
# ============================================================

# 허깅페이스 토큰 (https://huggingface.co/settings/tokens 에서 발급)
HUGGINGFACE_TOKEN = "허깅페이스 토큰 입력"  # 여기에 본인 토큰 입력

# 로컬 경로 설정
ADAPTER_PATH = "./gemma2-lora-adapters-kor"
MODEL_ID = "google/gemma-2-2b-it"

# **하드웨어 설정 (CPU로 고정)**
DEVICE = "cpu"
print(f"추론 장치 설정: {DEVICE.upper()} (CPU 사용)")

# 생성 파라미터
MAX_NEW_TOKENS = 150   # 생성할 최대 토큰(단어) 수 제한
TEMPERATURE = 0.7   # 답변의 창의성 제어
TOP_K = 50   # 샘플링에 고려할 가장 확률 높은 토큰 수 제한
TOP_P = 0.9   # 누적 확률 기준으로 샘플링에 고려할 토큰 집합 제한
REPETITION_PENALTY = 1.1   # 반복적인 단어 사용에 대한 패널티 부과: 1.0(패널티 없음)~ 1.5+(강한 패널티)

# ============================================================
# 2. 허깅페이스 로그인
# ============================================================

def login_huggingface():
    """
    허깅페이스에 로그인합니다.
    """
    print("허깅페이스 로그인 중...")
    
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not hf_token:
        # 환경 변수가 없으면 설정값 사용
        hf_token = HUGGINGFACE_TOKEN
    
    # **수정된 로직:** 토큰이 설정되지 않았을 때만 사용자에게 입력을 요구
    if not hf_token or hf_token.strip() == "" or hf_token == "your_huggingface_token_here":
        print("경고: HUGGINGFACE_TOKEN이 설정되지 않았습니다!")
        print("1. https://huggingface.co/settings/tokens 에서 토큰 생성")
        print("2. 위 코드의 HUGGINGFACE_TOKEN에 입력하거나")
        print("3. 환경 변수 HUGGINGFACE_TOKEN 설정")
        hf_token = input("토큰을 입력하세요: ")

    if not hf_token or hf_token.strip() == "":
        print("토큰 없이 로그인할 수 없습니다.")
        return False

    try:
        login(token=hf_token)
        print("허깅페이스 로그인 완료!")
        return True
    except Exception as e:
        print(f"로그인 실패: {e}")
        return False

# ============================================================
# 3. 모델 및 토크나이저 로드 (CPU 최적화)
# ============================================================

def load_model_and_tokenizer():
    """
    Gemma2 모델과 LoRA 어댑터를 로드합니다.
    """
    print("\n모델 로드 중 (CPU 환경에서는 매우 오래 걸릴 수 있습니다)...")
    
    compute_dtype = torch.float32 # CPU에서는 float32가 가장 안정적
    print(f"데이터 타입 설정: {compute_dtype}")
    
    # 토크나이저 로드
    print("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token # 패딩 토큰 설정
    
    # 원본 모델 로드 (CPU 명시적 할당)
    print("Gemma2 모델 로드 중...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
        ).to(DEVICE)
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print("   네트워크 연결, 허깅페이스 토큰, 모델 ID를 확인하세요.")
        return None, None
    
    # LoRA 어댑터 로드
    print("🔌 LoRA 어댑터 로드 중...")
    try:
        model_with_adapters = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        print("LoRA 어댑터 로드 완료!")
        model_with_adapters.eval()
    except Exception as e:
        print(f"어댑터 로드 실패: {e}")
        print(f"   경로 확인: {ADAPTER_PATH}")
        print("   Colab에서 다운받은 어댑터 폴더를 올바른 경로에 배치했는지 확인하세요.")
        return None, None
        
    return model_with_adapters, tokenizer

# ============================================================
# 4. 질문 처리 및 응답 생성 (Attention Mask 처리)
# ============================================================

def ask_question(model, tokenizer, question):
    """
    모델에 질문을 하고 답변을 생성합니다.
    """
    # 채팅 템플릿에 맞게 메시지 구성
    messages = [
        {"role": "user", "content": question}
    ]
    
    # 패딩 방향을 왼쪽으로 설정
    tokenizer.padding_side = "left"
    
    # 1. 채팅 템플릿을 문자열로 적용
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False, 
        add_generation_prompt=True,
    )

    # 2. 토크나이저 호출을 통해 input_ids와 attention_mask 모두 획득
    start_token_time = time.time()
    # **inputs 딕셔너리에 attention_mask가 포함됨**
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(DEVICE)
    end_token_time = time.time()
    
    # 답변 생성
    start_gen_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, # inputs 딕셔너리를 직접 전달 (input_ids와 attention_mask 포함)
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
        )
    end_gen_time = time.time()
    
    # 토큰 디코딩 (프롬프트 부분 제외)
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[-1]:], # 입력 토큰 길이 이후부터 디코딩
        skip_special_tokens=True
    )
    
    # 시간 정보 출력
    print(f"\n   - 토큰화 시간: {end_token_time - start_token_time:.2f}초")
    print(f"   - 답변 생성 시간: {end_gen_time - start_gen_time:.2f}초")
    
    return response.strip()

# ============================================================
# 5. 상호작용 인터페이스
# ============================================================

def interactive_chat(model, tokenizer):
    print("\n" + "="*60)
    print("Gemma2 + LoRA 어댑터 채팅 시작! (CPU 모드 - 응답이 매우 느릴 수 있습니다)")
    print("="*60)
    print("팁: 'exit' 또는 'quit'을 입력하면 종료됩니다.\n")
    
    conversation_count = 0
    
    while True:
        try:
            user_input = input("\n 질문: ").strip()
            
            if user_input.lower() in ['exit', 'quit', '종료']:
                print("\n 프로그램을 종료합니다.")
                break
            
            if not user_input:
                print(" 질문을 입력해주세요.")
                continue
            
            print("\n 답변 생성 중...", end="", flush=True)
            response = ask_question(model, tokenizer, user_input)
            
            print("\r" + " "*20 + "\r", end="")
            print(f" 답변: {response}")
            
            conversation_count += 1
            
        except KeyboardInterrupt:
            print("\n\n 프로그램을 종료합니다. (Ctrl+C)")
            break
        except Exception as e:
            print(f" 오류 발생: {e}")
            print("   다시 시도해주세요.\n")
    
    print(f"\n 총 {conversation_count}개의 질문을 처리했습니다.")

# ============================================================
# 6. 배치 질문 처리
# ============================================================

def batch_questions(model, tokenizer, questions):
    print("\n" + "="*60)
    print(" 배치 질문 처리 시작 (CPU 모드 - 응답이 매우 느릴 수 있습니다)")
    print("="*60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] 질문: {question}")
        response = ask_question(model, tokenizer, question)
        print(f"답변: {response}")

# ============================================================
# 7. 메인 함수 
# ============================================================

def main():
    print("="*60)
    print(f"Gemma2 + LoRA 어댑터 로컬 추론 시작 ({DEVICE.upper()} 기반)")
    print("="*60)
    
    # 1단계: 허깅페이스 로그인
    if not login_huggingface():
        print("로그인 실패. 프로그램을 종료합니다.")
        return
    
    # 2단계: 모델 및 어댑터 로드
    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        print("모델 로드 실패. 프로그램을 종료합니다.")
        return
    
    print("\n모든 준비 완료!")
    print("="*60)
    
    # 3단계: 인터페이스 선택
    print("\n사용 모드를 선택하세요:")
    print("1. 상호작용 채팅 (대화형)")
    print("2. 배치 질문 처리 (미리 정해진 질문들)")
    print("3. 종료")
    
    choice = input("\n선택 (1/2/3): ").strip()
    
    if choice == "1":
        interactive_chat(model, tokenizer)
    
    elif choice == "2":
        sample_questions = [
            "휴먼퓨처소프트는 언제 설립되었나요?",
            "휴먼퓨처소프트의 주요 제품은 무엇인가요?",
            "휴먼퓨처소프트에 대해 알려주세요.",
        ]
        batch_questions(model, tokenizer, sample_questions)
    
    elif choice == "3":
        print("프로그램을 종료합니다.")
    
    else:
        print("잘못된 선택입니다.")

# ============================================================
# 8. 프로그램 실행
# ============================================================

if __name__ == "__main__":
    main()