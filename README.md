# FairyTale Project 🌟

![GitHub Repo stars](https://img.shields.io/github/stars/sg-yun71/FairyTale?style=social)
![GitHub issues](https://img.shields.io/github/issues/sg-yun71/FairyTale)
![GitHub pull requests](https://img.shields.io/github/issues-pr/sg-yun71/FairyTale)
![License](https://img.shields.io/github/license/sg-yun71/FairyTale)

**FairyTale** 프로젝트는 AI 기반의 동화 생성 플랫폼으로, 다음 기능을 제공합니다:
- **사용자가 입력한 키워드와 연령대**를 기반으로 교훈적이고 자연스러운 동화를 생성합니다.
- 생성된 동화를 **삽화와 음성 파일**로 완성하여, 시각적 및 청각적 경험을 제공합니다.

---

## 📖 주요 기능

### 1. 맞춤형 동화 생성
- OpenAI GPT-4 기반의 LangChain 프레임워크를 사용하여, 키워드와 연령대에 적합한 동화를 생성합니다.

### 2. 삽화 제작
- Hugging Face의 AI 모델을 활용하여 동화 내용을 기반으로 한 삽화를 생성합니다.

### 3. 음성 변환 (TTS)
- Microsoft Edge TTS 엔진(`edge-tts`)을 사용하여 동화를 음성 파일(MP3)로 변환합니다.
- Pygame을 통해 생성된 음성을 재생할 수 있습니다.

### 4. 데이터베이스 검색
- Embedding을 통해 유사한 문서를 Chroma DB에서 검색하고, 문맥을 보완합니다.

### 5. 다국어 지원
- Google Translator API를 사용하여 동화를 다국어로 번역할 수 있습니다.

---

## 🛠️ 기술 스택

- **Python 3.9+**
- **LangChain**: 동화 생성 및 LLM 활용
- **edge-tts**: 음성 생성 및 변환
- **Pygame**: 오디오 파일 재생
- **Hugging Face**: AI 기반 삽화 제작
- **Chroma DB**: 문서 벡터화 및 검색
- **Pydub**: 오디오 파일 처리

---

## 🚀 설치 및 실행 방법

### 1. 저장소 복제

```bash
git clone https://github.com/sg-yun71/FairyTale.git
cd FairyTale