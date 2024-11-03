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


def load_environment():
    """환경 변수 로드 및 오류 처리"""
    try:
        load_dotenv('.env')
        if not os.getenv("OPENAI_API_KEY") or not os.getenv("HUGGINGFACE_API_KEY"):
            raise ValueError("API 키를 찾을 수 없습니다. .env 파일을 확인해 주세요.")
    except Exception as e:
        print(f"환경 변수 로드 중 오류 발생: {e}")
        exit()


def load_processed_data(category):
    """
    선택한 연령대에 해당하는 전처리된 JSON 데이터를 로드하는 함수
    :param category: 선택한 연령대 (유아, 초등_저학년, 초등_고학년)
    :return: 로드된 데이터
    """
    file_path = f"data/processed/{category}.json"
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        exit()
    except json.JSONDecodeError:
        print(f"JSON 파일을 디코딩하는 중 오류가 발생했습니다: {file_path}")
        exit()
    return data


def split_text(data, chunk_size=2000, chunk_overlap=500):
    """
    텍스트 데이터를 청크로 분할하는 함수
    :param data: 전처리된 데이터
    :param chunk_size: 청크 크기
    :param chunk_overlap: 청크 간 겹치는 부분
    :return: 분리된 텍스트 청크 리스트
    """
    rc_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n\n", "\n", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name="o200k_base",
        model_name="gpt-4o"
    )
    texts = [Document(page_content=story.get("description", "")) for story in data]
    text_documents = rc_text_splitter.split_documents(texts)
    return text_documents


def create_embedding_model():
    """
    임베딩 모델을 생성하는 함수
    :return: Hugging Face 임베딩 모델
    """
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return model


def embed_documents(docs, model, save_directory="./chroma_db"):
    """
    Embedding 모델을 사용하여 문서 임베딩 후 Chroma 벡터저장소(VectorStore)에 저장하는 함수
    :param docs: 분할된 문서
    :param model: 임베딩 모델
    :param save_directory: 벡터저장소 저장 경로
    :return: Chroma 데이터베이스 객체
    """
    import shutil
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 경고 억제

    # 기존 벡터 저장소 삭제
    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)

    db = Chroma.from_documents(docs, model, persist_directory=save_directory)
    return db


def create_llm(model_name="gpt-4", max_tokens=1500, temperature=0.7):
    """
    거대 언어 모델(LLM)을 생성하는 함수
    :return: 생성된 LLM 객체
    """
    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    return llm


def generate_story(llm, db, keywords):
    """
    키워드와 관련된 동화를 생성하는 함수
    :param llm: 거대 언어 모델
    :param db: Chroma 데이터베이스
    :param keywords: 동화 생성에 사용될 키워드 리스트
    :return: 생성된 동화 내용
    """
    query = ", ".join(keywords)

    try:
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 2, 'fetch_k': 3}
        )
        context_docs = retriever.invoke(query)  # 최신 메서드 invoke 사용
    except Exception as e:
        print(f"문서 검색 중 오류가 발생했습니다: {e}")
        return

    context_text = "\n".join(doc.page_content for doc in context_docs)
    max_context_length = 1000  # 컨텍스트 길이 축소
    if len(context_text) > max_context_length:
        context_text = context_text[:max_context_length] + "..."

    prompt_text = f"""
    당신은 어린이를 위한 동화를 만드는 AI입니다.
    주어진 키워드를 모두 포함하여 교훈적이고 자연스러운 이야기를 간결하게 작성하세요.
    동화는 다음 요소를 포함해야 합니다:
    - 이야기의 주제와 관련된 교훈을 중심으로 간결하게 진행합니다.
    - 아이들이 공감할 수 있는 간단한 대화와 상황을 담아 주세요.
    - 부적절한 소리 묘사나 과도하게 복잡한 내용은 피하고, 교육적이면서 흥미로운 내용을 담아 주세요.

    제목을 포함하여 동화 전체 내용을 간결하게 작성해 주세요.

    키워드: {keywords}
    참고 자료:
    {context_text}
    """

    try:
        response = llm.invoke([{"role": "system", "content": prompt_text}])
        story = response.content
        print(f"\n생성된 동화:\n{story}")
        return story
    except Exception as e:
        print(f"동화 생성 중 오류가 발생했습니다: {e}")
        return


def generate_illustrations(story):
    os.makedirs("illustrations", exist_ok=True)
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    # 스토리를 3단락으로 나눈 후 요약
    paragraphs = story.split("\n\n")[:3]
    prompts = []

    for i, paragraph in enumerate(paragraphs):
        prompt = (
            f"A fairy-tale style illustration depicting a whimsical scene based on: '{paragraph}'. "
            "Use vibrant colors and ensure consistent character appearance across all illustrations."
        )
        prompts.append(prompt)

    for i, prompt in enumerate(prompts, start=1):
        print(f"Generating illustration {i} with prompt: {prompt}")
        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/prompthero/openjourney",
                headers=headers,
                json={"inputs": prompt}
            )
            if response.status_code == 200:
                with open(f"illustrations/illustration_{i}.png", "wb") as f:
                    f.write(response.content)
                print(f"Generated illustration {i} saved as 'illustrations/illustration_{i}.png'")
            else:
                print(f"Error generating illustration {i}: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"삽화 생성 중 오류가 발생했습니다: {e}")


def run():
    print("환경 변수 로드 중...")
    load_environment()

    category = input("연령대를 선택하세요 (유아, 초등_저학년, 초등_고학년): ").strip()
    if category not in ["유아", "초등_저학년", "초등_고학년"]:
        print("잘못된 연령대입니다. 프로그램을 종료합니다.")
        return

    print("데이터를 로드 중입니다...")
    data = load_processed_data(category)
    print("데이터 분할 중입니다...")
    chunks = split_text(data)

    print("임베딩 모델 생성 중입니다...")
    embedding_model = create_embedding_model()

    print("벡터 데이터베이스 생성 중입니다...")
    db = embed_documents(chunks, embedding_model)

    print("LLM을 생성 중입니다...")
    llm = create_llm()

    keywords = input("동화의 주제나 요소가 될 키워드를 입력하세요 (콤마로 구분): ").split(",")

    print("동화를 생성 중입니다...")
    story = generate_story(llm, db, keywords)

    if story:
        print("삽화를 생성 중입니다...")
        generate_illustrations(story)
        print("모든 생성이 완료되었습니다!")


if __name__ == "__main__":
    run()