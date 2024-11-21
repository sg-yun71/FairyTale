import os
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

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
