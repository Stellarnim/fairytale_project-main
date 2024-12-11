import os
import pygame
import edge_tts
import re


# 음성 생성 및 저장 함수
async def generate_audio_from_text(text, filename, voice="ko-KR-SunHiNeural", output_format="mp3"):
    """
    텍스트를 음성으로 변환하고 MP3 파일로 저장
    """
    try:
        output_path = f"../static/audio/{filename}.{output_format}"  # 저장할 경로
        os.makedirs("audio", exist_ok=True)  # 폴더가 없으면 생성
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
async def generate_and_play_audio(story_title, story_paragraphs):
    """story_paragraphs = []

    story_paragraphs.append(story_title)
    for i, parts in enumerate(story_parts, start=0):
        story_paragraphs.append(parts)"""

    for i, paragraph in enumerate(story_paragraphs, start=1):
        dialogue = re.findall(r'"([^"]*)"', paragraph)

        # 대화가 포함된 문단은 남성 목소리로 설정
        if dialogue:
            voice = "ko-KR-InJoonNeural"  # 남성 목소리
        else:
            voice = "ko-KR-SunHiNeural"  # 여성 목소리

        filename = f"{story_title}_{i}"  # 저장할 파일 이름
        print(f"문단 {i}에 대한 음성 생성 중: {paragraph[:30]}...")  # 첫 30자 출력
        audio_path = await generate_audio_from_text(paragraph, filename, voice=voice)

        """if audio_path and os.path.exists(audio_path):  # 파일이 생성되었는지 확인
            print(f"문단 {i}에 대한 음성 재생 중...")
            play_audio(audio_path)
        else:
            print(f"문단 {i}에 대한 음성 생성 또는 파일 찾기에 실패했습니다.")"""
