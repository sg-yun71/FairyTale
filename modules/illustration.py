import os
import requests
import time
from deep_translator import GoogleTranslator

#스토리의 각 문단을 기반으로 삽화 생성
def generate_illustrations_from_story(story):
    """
    각 문단마다 품질 높은 삽화를 생성하는 함수
    """
    os.makedirs("illustrations", exist_ok=True)  # 삽화를 저장할 디렉토리 생성
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")  # Hugging Face API 키 로드
    headers = {"Authorization": f"Bearer {hf_api_key}"}  # 인증 헤더 설정

    # 스토리를 영어로 번역 후, 문단 단위로 분할
    translated_story = GoogleTranslator(source='ko', target='en').translate(story)
    story_paragraphs = translated_story.split("\n\n")  # 문단 단위로 분할

    for i, paragraph in enumerate(story_paragraphs, start=1):
        # 각 문단을 기반으로 한 삽화 생성 프롬프트 설정
        prompt = (
            f"Create a charming, detailed children's storybook illustration for the following scene. "
            f"The scene should be consistent with the overall fairy tale, focusing on creating a warm, friendly, and magical atmosphere. "
            f"Scene: {paragraph} "
            "Illustrate the emotions and interactions between the characters to reflect the story's narrative. "
            "Use soft pastel colors, gentle lighting, and simple yet inviting backgrounds, such as nature elements like clouds, trees, and rainbows. "
            "Ensure a cohesive storybook style across all illustrations. "
            "Do not include text in the image. Focus on visually telling the story through expressions and details that children can easily understand and connect with."
        )

        max_retries = 3  # 최대 재시도 횟수 설정
        retries = 0  # 현재 재시도 횟수 초기화
        success = False  # 성공 여부 플래그 초기화

        while not success and retries < max_retries:
            try:
                print(f"Generating illustration {i} with prompt: {prompt}")  # 삽화 생성 시작 메시지
                # Hugging Face API 호출하여 이미지 생성
                response = requests.post(
                    "https://api-inference.huggingface.co/models/Shakker-Labs/FLUX.1-dev-LoRA-One-Click-Creative-Template",
                    headers=headers,
                    json={"inputs": prompt}
                )

                if response.status_code == 200:  # 이미지 생성 성공 시
                    with open(f"illustrations/story_illustration_{i}.png", "wb") as f:
                        f.write(response.content)  # 이미지를 파일로 저장
                    print(f"Generated illustration saved as 'illustrations/story_illustration_{i}.png'")
                    success = True  # 성공 플래그 업데이트
                elif response.status_code == 503:  # 모델 로딩 중인 경우
                    print("Model is loading; retrying in 20 seconds...")
                    time.sleep(20)
                elif response.status_code == 500:  # 서버 에러 발생 시
                    print("Server error encountered; retrying in 5 seconds...")
                    time.sleep(5)
                elif response.status_code == 429:  # 요청 제한 초과 시
                    print("Request limit reached; waiting for 1 minute before retrying...")
                    time.sleep(60)
                else:
                    print(f"Error generating illustration: {response.status_code} - {response.text}")
                    break  # 기타 오류 발생 시 반복 종료
            except Exception as e:
                print(f"Illustration generation error for part {i}: {e}")  # 예외 발생 시 에러 메시지 출력
            retries += 1  # 재시도 횟수 증가

        if not success:
            print(f"Failed to generate illustration {i} after {max_retries} attempts.")  # 최대 재시도 횟수 초과 시 실패 메시지 출력

        time.sleep(1)  # 다음 요청 전 대기