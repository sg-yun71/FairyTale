import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 (API 키 등)를 로드하는 함수
def load_environment():
    print("환경 변수 로드 중...")  # 로딩 메시지 출력
    load_dotenv('.env')  # .env 파일에서 환경 변수를 로드
    # 환경 변수 OPENAI_API_KEY와 HUGGINGFACE_API_KEY가 있는지 확인
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("HUGGINGFACE_API_KEY"):
        raise ValueError("API 키를 찾을 수 없습니다. .env 파일을 확인해 주세요.")  # 키가 없을 경우 에러 발생
    print("환경 변수 로드 완료")  # 로딩 완료 메시지 출력