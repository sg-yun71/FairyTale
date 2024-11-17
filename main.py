import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import requests
from deep_translator import GoogleTranslator
import time
from pydub import AudioSegment
import pygame
import edge_tts
import asyncio


# .env 파일에서 환경 변수 (API 키 등)를 로드하는 함수
def load_environment():
    print("환경 변수 로드 중...")  # 로딩 메시지 출력
    load_dotenv('.env')  # .env 파일에서 환경 변수를 로드
    # 환경 변수 OPENAI_API_KEY와 HUGGINGFACE_API_KEY가 있는지 확인
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("HUGGINGFACE_API_KEY"):
        raise ValueError("API 키를 찾을 수 없습니다. .env 파일을 확인해 주세요.")  # 키가 없을 경우 에러 발생
    print("환경 변수 로드 완료")  # 로딩 완료 메시지 출력

# 주어진 카테고리에 맞는 JSON 파일을 로드하여 데이터 반환
def load_processed_data(category):
    file_path = f"data/processed/{category}.json"  # 카테고리에 맞는 파일 경로 설정
    print(f"파일 로드 중: {file_path}")  # 파일 로드 중 메시지 출력
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # JSON 파일 로드
    print("파일 로드 완료")  # 파일 로드 완료 메시지 출력
    return data  # 데이터 반환

# 텍스트 데이터를 분할하여 문서 객체 리스트로 반환
def split_text(data, chunk_size=2000, chunk_overlap=500):
    print("데이터 분할 중...")  # 분할 시작 메시지 출력
    # 텍스트 분할기 생성, 설정에 따라 데이터 분할
    rc_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n\n", "\n", " "],  # 분할 기준 (새 줄, 공백 등)
        chunk_size=chunk_size,  # 분할할 청크 크기
        chunk_overlap=chunk_overlap,  # 청크 간 중첩 부분
        encoding_name="o200k_base",
        model_name="gpt-4o"
    )
    # 각 데이터 항목을 Document 객체로 변환하여 리스트 생성
    texts = [Document(page_content=story.get("description", "")) for story in data]
    text_documents = rc_text_splitter.split_documents(texts)  # Document 리스트를 분할
    print("데이터 분할 완료")  # 분할 완료 메시지 출력
    return text_documents  # 분할된 텍스트 반환

# 임베딩 모델 생성 및 반환
def create_embedding_model():
    print("임베딩 모델 생성 중...")
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    print("임베딩 모델 생성 완료")
    return model

# 문서 데이터를 벡터화하여 데이터베이스에 저장
def embed_documents(docs, model, save_directory="./chroma_db"):
    print("벡터 데이터베이스 생성 중...")  # 데이터베이스 생성 시작 메시지 출력
    import shutil
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 병렬 토크나이저 설정 비활성화
    if os.path.exists(save_directory):  # 기존 데이터베이스 디렉토리 삭제
        shutil.rmtree(save_directory)
    # 문서 리스트를 벡터화하여 데이터베이스에 저장
    db = Chroma.from_documents(docs, model, persist_directory=save_directory)
    print("벡터 데이터베이스 생성 완료")  # 데이터베이스 생성 완료 메시지 출력
    return db  # 벡터 데이터베이스 반환

# 언어 모델 (LLM) 생성 함수
def create_llm(model_name="gpt-4", max_tokens=1500, temperature=0.7):
    print("LLM 생성 중...")  # 모델 생성 시작 메시지 출력
    # ChatOpenAI 모델 생성, 설정에 맞춰서 생성
    llm = ChatOpenAI(model_name=model_name, openai_api_key=os.getenv("OPENAI_API_KEY"),
                     temperature=temperature, max_tokens=max_tokens, streaming=True,
                     callbacks=[StreamingStdOutCallbackHandler()])
    print("LLM 생성 완료")  # 모델 생성 완료 메시지 출력
    return llm  # 생성된 LLM 반환

# 주어진 키워드로부터 동화 생성
def generate_story(llm, db, keywords):
    print("동화 생성 중...")  # 동화 생성 시작 메시지 출력
    query = ", ".join(keywords)  # 키워드를 문자열로 연결하여 쿼리 생성
    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 2, 'fetch_k': 3})  # 데이터베이스에서 검색을 위한 리트리버 생성
    context_docs = retriever.invoke(query)  # 리트리버를 사용하여 컨텍스트 문서 검색
    context_text = "\n".join(doc.page_content for doc in context_docs)[:1000]  # 검색 결과를 텍스트로 변환
    # 동화 생성 프롬프트 텍스트 설정
    prompt_text = f"""
    당신은 어린이를 위한 동화를 만드는 AI입니다. 주어진 키워드를 모두 포함하여 교훈적이고 자연스러운 이야기를 간결하게 작성하세요.
    동화는 다음 요소를 포함해야 합니다:
    - 이야기의 주제와 관련된 교훈을 중심으로 간결하게 진행합니다.
    - 아이들이 공감할 수 있는 간단한 대화와 상황을 담아 주세요.
    - 부적절한 소리 묘사나 과도하게 복잡한 내용은 피하고, 교육적이면서 흥미로운 내용을 담아 주세요.

    제목을 포함하여 동화 전체 내용을 간결하게 작성해 주세요.

    키워드: {keywords}
    참고 자료:
    {context_text}
    """
    # LLM에 프롬프트 전달하여 동화 생성
    response = llm.invoke([{"role": "system", "content": prompt_text}])
    return response.content  # 생성된 동화 반환

#스토리의 각 문단을 기반으로 삽화 생성
# def generate_illustrations_from_story(story):
#     """
#     각 문단마다 품질 높은 삽화를 생성하는 함수
#     """
#     os.makedirs("illustrations", exist_ok=True)  # 삽화를 저장할 디렉토리 생성
#     hf_api_key = os.getenv("HUGGINGFACE_API_KEY")  # Hugging Face API 키 로드
#     headers = {"Authorization": f"Bearer {hf_api_key}"}  # 인증 헤더 설정
#
#     # 스토리를 영어로 번역 후, 문단 단위로 분할
#     translated_story = GoogleTranslator(source='ko', target='en').translate(story)
#     story_paragraphs = translated_story.split("\n\n")  # 문단 단위로 분할
#
#     for i, paragraph in enumerate(story_paragraphs, start=1):
#         # 각 문단을 기반으로 한 삽화 생성 프롬프트 설정
#         prompt = (
#             f"Create a charming, detailed children's storybook illustration for the following scene. "
#             f"The scene should be consistent with the overall fairy tale, focusing on creating a warm, friendly, and magical atmosphere. "
#             f"Scene: {paragraph} "
#             "Illustrate the emotions and interactions between the characters to reflect the story's narrative. "
#             "Use soft pastel colors, gentle lighting, and simple yet inviting backgrounds, such as nature elements like clouds, trees, and rainbows. "
#             "Ensure a cohesive storybook style across all illustrations. "
#             "Do not include text in the image. Focus on visually telling the story through expressions and details that children can easily understand and connect with."
#         )
#
#         max_retries = 3  # 최대 재시도 횟수 설정
#         retries = 0  # 현재 재시도 횟수 초기화
#         success = False  # 성공 여부 플래그 초기화
#
#         while not success and retries < max_retries:
#             try:
#                 print(f"Generating illustration {i} with prompt: {prompt}")  # 삽화 생성 시작 메시지
#                 # Hugging Face API 호출하여 이미지 생성
#                 response = requests.post(
#                     "https://api-inference.huggingface.co/models/Shakker-Labs/FLUX.1-dev-LoRA-One-Click-Creative-Template",
#                     headers=headers,
#                     json={"inputs": prompt}
#                 )
#
#                 if response.status_code == 200:  # 이미지 생성 성공 시
#                     with open(f"illustrations/story_illustration_{i}.png", "wb") as f:
#                         f.write(response.content)  # 이미지를 파일로 저장
#                     print(f"Generated illustration saved as 'illustrations/story_illustration_{i}.png'")
#                     success = True  # 성공 플래그 업데이트
#                 elif response.status_code == 503:  # 모델 로딩 중인 경우
#                     print("Model is loading; retrying in 20 seconds...")
#                     time.sleep(20)
#                 elif response.status_code == 500:  # 서버 에러 발생 시
#                     print("Server error encountered; retrying in 5 seconds...")
#                     time.sleep(5)
#                 elif response.status_code == 429:  # 요청 제한 초과 시
#                     print("Request limit reached; waiting for 1 minute before retrying...")
#                     time.sleep(60)
#                 else:
#                     print(f"Error generating illustration: {response.status_code} - {response.text}")
#                     break  # 기타 오류 발생 시 반복 종료
#             except Exception as e:
#                 print(f"Illustration generation error for part {i}: {e}")  # 예외 발생 시 에러 메시지 출력
#             retries += 1  # 재시도 횟수 증가
#
#         if not success:
#             print(f"Failed to generate illustration {i} after {max_retries} attempts.")  # 최대 재시도 횟수 초과 시 실패 메시지 출력
#
#         time.sleep(1)  # 다음 요청 전 대기


# 음성 생성 및 저장 함수
async def generate_audio_from_text(text, filename, voice="ko-KR-SunHiNeural", output_format="mp3"):
    try:
        output_path = f"audio/{filename}.{output_format}"
        os.makedirs("audio", exist_ok=True)
        communicate = edge_tts.Communicate(text, voice=voice)
        await communicate.save(output_path)
        print(f"음성 파일 생성 완료: {output_path}")
        return output_path
    except Exception as e:
        print(f"음성 생성 실패: {e}")
        return None


# MP3 파일 재생 함수
def play_audio(file_path):
    print(f"재생 중: {file_path}")
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pass  # 음악이 끝날 때까지 대기

    print("재생 완료.")


# 문단별 음성 생성 및 재생 함수
async def generate_and_play_audio(story_paragraphs):
    for i, paragraph in enumerate(story_paragraphs, start=1):
        filename = f"story_paragraph_{i}"
        print(f"Generating audio for paragraph {i}: {paragraph[:30]}...")
        audio_path = await generate_audio_from_text(paragraph, filename)

        if audio_path and os.path.exists(audio_path):
            print(f"Playing audio for paragraph {i}...")
            play_audio(audio_path)
        else:
            print(f"Failed to generate or find audio for paragraph {i}.")

# 프로그램 실행 함수
def run():
    print("프로그램 시작")
    load_environment()  # 환경 변수 로드
    category = input("연령대를 선택하세요 (유아, 초등_저학년, 초등_고학년): ").strip()  # 사용자 입력으로 연령대 선택
    data = load_processed_data(category)  # 선택한 연령대에 맞는 데이터 로드
    chunks = split_text(data)  # 데이터 분할
    embedding_model = create_embedding_model()  # 임베딩 모델 생성
    db = embed_documents(chunks, embedding_model)  # 데이터베이스에 문서 임베딩
    llm = create_llm()  # LLM 생성
    keywords = input("동화의 주제나 요소가 될 키워드를 입력하세요 (콤마로 구분): ").split(",")  # 키워드 입력
    story = generate_story(llm, db, keywords)  # 동화 생성

    if story:
        #generate_illustrations_from_story(story)  # 동화의 각 부분을 기반으로 삽화 생성

        # 오디오 생성 폴더 준비
        os.makedirs("audio", exist_ok=True)  # 오디오 폴더 준비

        # 스토리를 문단 단위로 나누고 오디오 생성 및 재생
        story_paragraphs = story.split("\n\n")
        asyncio.run(generate_and_play_audio(story_paragraphs))  # 비동기 호출

        print("모든 생성이 완료되었습니다!")  # 생성 완료 메시지 출력


# 프로그램 메인 실행
if __name__ == "__main__":
    run()