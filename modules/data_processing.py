import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

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