import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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