import os
import pygame
import edge_tts
import asyncio

# 음성 생성 및 저장 함수
async def generate_audio_from_text(text, filename, voice="ko-KR-SunHiNeural", output_format="mp3"):
    """
        텍스트를 음성으로 변환하고 MP3 파일로 저장
        """
    try:
        output_path = f"audio/{filename}.{output_format}" # 저장할 경로
        os.makedirs("audio", exist_ok=True) # 폴더가 없으면 생성
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
        filename = f"story_paragraph_{i}" # 저장할 파일 이름
        print(f"Generating audio for paragraph {i}: {paragraph[:30]}...") # 첫 30자 출력
        audio_path = await generate_audio_from_text(paragraph, filename)

        if audio_path and os.path.exists(audio_path): # 파일이 생성되었는지 확인
            print(f"Playing audio for paragraph {i}...")
            play_audio(audio_path)
        else:
            print(f"Failed to generate or find audio for paragraph {i}.")
