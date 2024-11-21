from modules.environment import load_environment
from modules.data_processing import load_processed_data, split_text
from modules.embedding import create_embedding_model, embed_documents
from modules.story_generation import create_llm, generate_story
from modules.illustration import generate_illustrations_from_story
from modules.tts import generate_and_play_audio
import asyncio


def run():
    print("프로그램 시작")
    load_environment()  # 환경 변수 로드

    # 연령대 선택 및 데이터 로드
    category = input("연령대를 선택하세요 (유아, 초등_저학년, 초등_고학년): ").strip()
    data = load_processed_data(category)

    # 데이터 분할 및 임베딩 처리
    chunks = split_text(data)
    embedding_model = create_embedding_model()
    db = embed_documents(chunks, embedding_model)

    # 키워드 기반 동화 생성
    llm = create_llm()
    keywords = input("동화의 주제나 요소가 될 키워드를 입력하세요 (콤마로 구분): ").split(",")
    story = generate_story(llm, db, keywords)

    if story:
        # 삽화 생성
        generate_illustrations_from_story(story)

        # 오디오 생성 및 재생
        story_paragraphs = story.split("\n\n")
        asyncio.run(generate_and_play_audio(story_paragraphs))

        print("모든 생성이 완료되었습니다!")  # 완료 메시지


if __name__ == "__main__":
    run()
