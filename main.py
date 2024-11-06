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

def load_environment():
    print("환경 변수 로드 중...")
    load_dotenv('.env')
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("HUGGINGFACE_API_KEY"):
        raise ValueError("API 키를 찾을 수 없습니다. .env 파일을 확인해 주세요.")
    print("환경 변수 로드 완료")

def load_processed_data(category):
    file_path = f"data/processed/{category}.json"
    print(f"파일 로드 중: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print("파일 로드 완료")
    return data

def split_text(data, chunk_size=2000, chunk_overlap=500):
    print("데이터 분할 중...")
    rc_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n\n", "\n", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name="o200k_base",
        model_name="gpt-4o"
    )
    texts = [Document(page_content=story.get("description", "")) for story in data]
    text_documents = rc_text_splitter.split_documents(texts)
    print("데이터 분할 완료")
    return text_documents

def create_embedding_model():
    print("임베딩 모델 생성 중...")
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    print("임베딩 모델 생성 완료")
    return model

def embed_documents(docs, model, save_directory="./chroma_db"):
    print("벡터 데이터베이스 생성 중...")
    import shutil
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)
    db = Chroma.from_documents(docs, model, persist_directory=save_directory)
    print("벡터 데이터베이스 생성 완료")
    return db

def create_llm(model_name="gpt-4", max_tokens=1500, temperature=0.7):
    print("LLM 생성 중...")
    llm = ChatOpenAI(model_name=model_name, openai_api_key=os.getenv("OPENAI_API_KEY"),
                     temperature=temperature, max_tokens=max_tokens, streaming=True,
                     callbacks=[StreamingStdOutCallbackHandler()])
    print("LLM 생성 완료")
    return llm

def generate_story(llm, db, keywords):
    print("동화 생성 중...")
    query = ", ".join(keywords)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 2, 'fetch_k': 3})
    context_docs = retriever.invoke(query)
    context_text = "\n".join(doc.page_content for doc in context_docs)[:1000]
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
    response = llm.invoke([{"role": "system", "content": prompt_text}])
    return response.content

def generate_illustrations_from_story(story):
    """
    각 문단마다 품질 높은 삽화를 생성하는 함수
    """
    os.makedirs("illustrations", exist_ok=True)
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    # 스토리를 영어로 번역 후, 문단 단위로 분할
    translated_story = GoogleTranslator(source='ko', target='en').translate(story)
    story_paragraphs = translated_story.split("\n\n")

    for i, paragraph in enumerate(story_paragraphs, start=1):
        prompt = (
            f"Create a charming, detailed children's storybook illustration for the following scene. "
            f"The scene should be consistent with the overall fairy tale, focusing on creating a warm, friendly, and magical atmosphere. "
            f"Scene: {paragraph} "
            "Illustrate the emotions and interactions between the characters to reflect the story's narrative. "
            "Use soft pastel colors, gentle lighting, and simple yet inviting backgrounds, such as nature elements like clouds, trees, and rainbows. "
            "Ensure a cohesive storybook style across all illustrations. "
            "Do not include text in the image. Focus on visually telling the story through expressions and details that children can easily understand and connect with."
        )

        max_retries = 3
        retries = 0
        success = False

        while not success and retries < max_retries:
            try:
                print(f"Generating illustration {i} with prompt: {prompt}")
                response = requests.post(
                    "https://api-inference.huggingface.co/models/Shakker-Labs/FLUX.1-dev-LoRA-One-Click-Creative-Template",
                    headers=headers,
                    json={"inputs": prompt}
                )

                if response.status_code == 200:
                    with open(f"illustrations/story_illustration_{i}.png", "wb") as f:
                        f.write(response.content)
                    print(f"Generated illustration saved as 'illustrations/story_illustration_{i}.png'")
                    success = True
                elif response.status_code == 503:
                    print("Model is loading; retrying in 20 seconds...")
                    time.sleep(20)
                elif response.status_code == 500:
                    print("Server error encountered; retrying in 5 seconds...")
                    time.sleep(5)
                elif response.status_code == 429:
                    print("Request limit reached; waiting for 1 minute before retrying...")
                    time.sleep(60)
                else:
                    print(f"Error generating illustration: {response.status_code} - {response.text}")
                    break
            except Exception as e:
                print(f"Illustration generation error for part {i}: {e}")
            retries += 1

        if not success:
            print(f"Failed to generate illustration {i} after {max_retries} attempts.")

        time.sleep(1)

def run():
    print("프로그램 시작")
    load_environment()
    category = input("연령대를 선택하세요 (유아, 초등_저학년, 초등_고학년): ").strip()
    data = load_processed_data(category)
    chunks = split_text(data)
    embedding_model = create_embedding_model()
    db = embed_documents(chunks, embedding_model)
    llm = create_llm()
    keywords = input("동화의 주제나 요소가 될 키워드를 입력하세요 (콤마로 구분): ").split(",")
    story = generate_story(llm, db, keywords)
    if story:
        generate_illustrations_from_story(story)
        print("모든 생성이 완료되었습니다!")

if __name__ == "__main__":
    run()