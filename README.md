# 인공지능 자동 자막 생성기 (AI Auto Video Subtitles Generator)

음성 또는 영상 파일에서 자동으로 자막을 생성하고, 선택적으로 LLM(대규모 언어 모델)을 사용하여 자막을 교정하는 도구입니다.

## YouTube

[![자동 자막 생성기 데모](https://img.youtube.com/vi/ylGXAXliyjE/mqdefault.jpg)](https://youtu.be/ylGXAXliyjE)

*클릭하면 YouTube로 이동합니다*

## 자막 생성 과정

![자막 생성 과정](./process.png)

## 주요 기능

- **한글 지원**: 설치부터 실사용까지 모든 과정에서 친절한 한글 안내 표시
- **원클릭 설치**: 설치 파일 실행 한 번으로 설치과정 끝
- **편리한 UI**: Streamlit으로 구현되어 사용하기 편리한 웹 UI 제공
- **로컬PC 구동**: 인터넷 연결 없이도 자막 생성 기능 작동(설치 과정에서는 인터넷 연결 필요)
- **OpenAI Whisper**: 원하는 정확성과 속도에 따라 5가지 STT 모델(small, large 등) 중에서 선택 가능
- **다양한 형식 지원**: mp3, wav, mp4, avi, mov, mkv 등 다양한 오디오/비디오 파일 형식 지원
- **다국어 지원**: 한국어, 영어, 일본어, 중국어 등 자동 언어 감지 및 지정 기능
- **자막 미리보기**: 생성되는 자막을 실시간으로 확인 가능
- **다운로드 지원**: SRT 및 VTT 형식으로 자막 다운로드
- **LLM 교정**: OpenAI 또는 Anthropic Claude API를 활용한 자막 교정 기능(옵션)
- **VAD(Voice Activity Detection)**: 음성이 있는 부분만 감지하여 처리하는 기능(옵션). OpenAI Whisper 모델의 잘 알려진 버그로 인해 음성 공백 구간이 긴 영상 및 음성 파일을 처리할 때 특히 유용

![자동 자막 생성기 스크린샷](./screenshot.png)

## 시스템 요구사항

- **운영체제**: Windows, macOS, 또는 Linux
- **Python**: 3.8 이상 (3.10 권장, 3.13에서는 일부 호환성 문제가 있을 수 있음)
- **디스크 공간**: 최소 2GB (Whisper 모델 크기에 따라 증가할 수 있음)
- **RAM**: 최소 4GB (large 모델 사용 시 8GB 이상 권장)
- **선택적 요구사항**:
  - **Microsoft Visual C++ 빌드 도구**: webrtcvad 모듈 사용 시 필요 (자동 설치 지원)
  - **FFmpeg**: 오디오/비디오 처리에 필요 (자동 설치 지원)
- **인터넷 연결**: 초기 설치 및 LLM 교정 API 사용 시 필요
  - 최초 설치 및 실행 이후에는 인터넷 연결 없이 **로컬로도 구동 가능**
  - **인터넷이 필요한 기능(옵션)**: LLM 교정

## 설치 방법

※ **윈도우 PC**에서만 테스트되었습니다. 맥 또는 리눅스에서는 테스트되지 않았기 때문에 설치를 보장하지 못합니다.

### Windows

1. 이 레포지토리를 다운로드 또는 클론합니다.
   - 다운로드보단 git clone을 추천드립니다.
   - 다운로드한 경우 통합설치&실행파일(`install_run_windows.ps1`) 실행 시 '이 파일을 열기 전에 항상 확인' 체크박스를 해제해야 합니다.
2. `install_run_windows.ps1` 통합설치&실행파일을 마우스 오른쪽 버튼 클릭 후, "PowerShell에서 실행"을 클릭합니다.
   - PowerShell에서 실행 정책 제한이 있는 경우: `powershell -ExecutionPolicy Bypass -File "install_run_windows.ps1"`
3. 화면의 지시에 따라 'Y'를 입력하며 설치를 진행합니다.
   - 파이썬이 설치되어 있지 않은 경우 자동으로 Python 3.10을 설치할 수 있습니다.
   - Microsoft Visual C++ 빌드 도구가 없는 경우 자동으로 설치하거나 이 기능을 건너뛸 수 있습니다.
   - FFmpeg도 선택적으로 자동 설치할 수 있습니다.
   - 필요한 라이브러리가 자동으로 설치됩니다.
4. 모든 설치가 완료되면 자동으로 실행됩니다.
5. **이후 다시 실행할 때는 `자막생성기.bat` 파일을 더블클릭**하면 됩니다.

### macOS/Linux

1. 이 레포지토리를 다운로드 또는 클론합니다.
2. 터미널에서 프로젝트 폴더로 이동합니다.
3. 다음 명령을 실행합니다:
   ```bash
   chmod +x "1. install_mac_linux.sh"
   ./1.\ install_mac_linux.sh
   ```
4. 화면의 지시에 따라 설치를 진행합니다.

## 사용 방법

### 실행

- **Windows**: `자막생성기.bat` 파일을 더블클릭하여 실행합니다.
  - 또는 `install_run_windows.ps1` 파일을 우클릭한 후, "PowerShell에서 실행"을 눌러 실행합니다.
  - 바탕화면에서 바로 실행하려면: `자막생성기.bat` 파일을 우클릭 → "바로 가기 만들기" → 바로가기를 바탕화면으로 이동
- **macOS/Linux**: 터미널에서 `./2. run_mac_linux.sh` 명령을 실행합니다.

실행 후 웹 브라우저가 자동으로 열리며 Streamlit 인터페이스가 표시됩니다. 브라우저가 자동으로 열리지 않는 경우 `http://localhost:8501`로 접속하세요.

### 기본 사용법

1. 사이드바에서 모델 설정 및 자막 생성 옵션을 선택합니다.
2. "파일 업로드" 버튼을 클릭하여 음성 또는 영상 파일을 업로드합니다.
3. "자막 생성 시작" 버튼을 클릭합니다.
4. 자막 생성이 완료되면 SRT 또는 VTT 형식으로 다운로드할 수 있습니다.

### 주요 설정

#### Whisper 모델 크기
- **tiny**: 가장 빠르지만 정확도 낮음
- **base**: 빠르고 기본적인 정확도
- **small**: 적절한 속도와 정확도 (기본 권장)
- **medium**: 높은 정확도, 느린 속도
- **large**: 가장 높은 정확도, 가장 느린 속도

#### LLM 교정 제공자
- **사용안함**: LLM 교정 없이 Whisper 결과 그대로 사용
- **OpenAI**: OpenAI API를 사용하여 자막 교정 (API 키 필요)
  - 사용 모델: `gpt-5-mini`
- **Anthropic**: Anthropic Claude API를 사용하여 자막 교정 (API 키 필요)
  - 사용 모델: `claude-haiku-4-5`

#### API 키 설정
OpenAI 또는 Anthropic을 선택한 경우, "API 키 설정" 확장 메뉴에서 API 키를 입력해야 합니다. 또는 프로젝트 루트 폴더에 `.env` 파일을 생성하고 다음과 같이 API 키를 설정할 수 있습니다:

```
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

#### 자막 옵션
- **자막 언어**: 자동 감지 또는 한국어, 영어, 일본어, 중국어 중 선택
- **최대/최소 글자 수**: 한 자막당 표시할 최대/최소 글자 수
- **최대 시간**: 한 자막당 최대 지속 시간 (초)
- **VAD(Voice Activity Detection)**: 음성이 있는 부분만 감지하여 처리 (선택적 기능, Visual C++ 빌드 도구 필요). OpenAI Whisper 모델의 잘 알려진 버그로 인해 음성 공백 구간이 긴 영상 및 음성 파일을 처리할 때 특히 중요

#### 업로드 크기 제한

기본적으로 파일 업로드 크기 제한은 1GB(1000MB)로 설정되어 있습니다. 더 큰 파일을 처리하거나 제한을 변경하려면 다음과 같이 할 수 있습니다:

1. **설정 파일 수정**:
   - `.streamlit/config.toml` 파일을 열고 다음 값을 원하는 크기로 변경합니다:
     ```toml
     [server]
     maxUploadSize = 2000  # 원하는 크기(MB)로 변경 (예: 2GB)
     ```

2. **명령줄에서 실행하는 방법**:
   - 임시로 다른 값을 적용하려면 실행 파일에 매개변수를 추가합니다:
     ```
     # Windows
     2. run_windows.bat --server.maxUploadSize=2000
     
     # macOS/Linux
     ./2. run_mac_linux.sh --server.maxUploadSize=2000
     ```

대용량 비디오 파일의 경우, 자막 생성 전에 오디오만 추출하여 MP3로 변환하면 처리 속도도 빨라질 수 있습니다.

## 문제 해결

### 일반적인 문제

1. **"Python is not recognized as an internal or external command"**
   - Python이 PATH에 추가되지 않았습니다. Python을 재설치하고 "Add to PATH" 옵션을 선택하세요.

2. **패키지 설치 오류**
   - 인터넷 연결을 확인하세요.
   - 방화벽이나 프록시가 설치를 차단하지 않는지 확인하세요.
   - 관리자 권한으로 실행해보세요.

3. **"error: Microsoft Visual C++ 14.0 or greater is required"**
   - Microsoft Visual C++ 빌드 도구가 필요합니다. 설치 스크립트를 다시 실행하여 자동으로 설치하거나, 수동으로 설치하세요.
   - 또는 VAD 기능 사용을 포기하고 webrtcvad 모듈 없이 계속 진행할 수 있습니다.

4. **CUDA 관련 오류**
   - 리눅스에서 CUDA 오류가 발생할 경우: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` 명령으로 CPU 버전의 PyTorch를 설치하세요.

5. **웹브라우저가 자동으로 열리지 않는 경우**
   - 브라우저를 수동으로 열고 `http://localhost:8501` 주소로 접속하세요.

6. **"Module not found" 오류**
   - 프로그램 폴더로 이동하여 설치 스크립트를 다시 실행하세요.

7. **Python 3.13 호환성 문제**
   - Python 3.13에서 일부 패키지와 호환성 문제가 발생할 수 있습니다.
   - Python 3.10으로 다운그레이드하거나, 가상환경에 Python 3.10을 사용하는 것을 권장합니다.

### LLM 교정 관련 문제

1. **"API key not provided" 오류**
   - API 키가 올바르게 입력되었는지 확인하세요.
   - OpenAI 또는 Anthropic 계정에 잔액이 있는지 확인하세요.

2. **자막 교정이 느린 경우**
   - OpenAI, Anthropic 등 LLM API 서버 요청이 많아 느린 경우가 대다수입니다.
   - 자막 길이를 짧게 설정하여 시도해보세요.

## 주의사항

- 매우 긴 오디오 파일(1시간 이상)은 처리에 시간이 많이 소요될 수 있습니다.
- LLM 교정 사용 시 API 사용 비용이 발생할 수 있습니다.
- Whisper 모델은 첫 실행 시 다운로드되므로 인터넷 연결이 필요합니다.
- VAD 기능은 선택적이지만, 음성 공백 구간이 긴 영상이나 음성 파일을 처리할 때 매우 중요합니다. OpenAI Whisper 모델은 긴 무음 구간에서 오작동하는 알려진 버그가 있어 VAD가 이 문제를 해결합니다. 하지만 Microsoft Visual C++ 빌드 도구 없이도 기본 자막 생성 기능은 사용할 수 있습니다.

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.

## 기여

이슈 및 풀 리퀘스트를 환영합니다. 기여하기 전에 이슈를 통해 변경 사항을 논의해주세요.

## 제작 정보

이 프로그램은 [메타마인드](https://metamind.kr)가 제작하였으며, 자유로운 수정 및 공유가 가능합니다.

---

© 2025 메타마인드 | [https://metamind.kr](https://metamind.kr)