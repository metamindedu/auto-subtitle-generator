"""
자동 자막 생성기 - 메인 애플리케이션

음성 또는 영상 파일을 업로드하여 자동으로 자막을 생성합니다.
Whisper 모델을 사용하여 음성을 인식하고, 선택적으로 LLM을 사용하여 자막을 교정합니다.
"""

import warnings
import streamlit as st
from dotenv import load_dotenv

from src.config.api_keys import load_saved_api_keys
from src.utils.vad_utils import is_vad_available
from src.ui.sidebar import render_sidebar
from src.ui.main_content import (
    render_subtitle_history,
    render_subtitle_preview,
    render_file_upload,
    render_footer
)

# .env 파일 로드
load_dotenv()

# 경고 메시지 필터링
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", category=FutureWarning)


def init_session_state():
    """세션 상태 초기화"""
    # VAD 모듈 로드 상태
    if 'vad_module_loaded' not in st.session_state:
        st.session_state.vad_module_loaded = is_vad_available()
        if not st.session_state.vad_module_loaded:
            st.warning("webrtcvad 모듈이 설치되지 않았습니다. 'pip install webrtcvad'로 설치하세요. VAD 없이 계속 진행합니다.")

    # 저장된 API 키 로드
    if 'api_keys_loaded' not in st.session_state:
        saved_keys = load_saved_api_keys()
        st.session_state.openai_api_key = saved_keys.get('openai_api_key', '')
        st.session_state.anthropic_api_key = saved_keys.get('anthropic_api_key', '')
        st.session_state.save_api_keys_enabled = bool(saved_keys)
        st.session_state.api_keys_loaded = True

    # 기본 세션 상태 초기화
    defaults = {
        'openai_api_key': '',
        'anthropic_api_key': '',
        'save_api_keys_enabled': False,
        'last_srt_content': None,
        'last_filename': None,
        'show_last_preview': False,
        'correction_logs': [],
        'subtitle_history': [],
        'correction_logs_history': {},
        'show_logs': False
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def main():
    """메인 함수"""
    # 페이지 설정
    st.set_page_config(
        page_title="자동 자막 생성기",
        page_icon="🎬",
        layout="wide"
    )

    # 세션 상태 초기화
    init_session_state()

    # 사이드바 렌더링 및 설정 가져오기
    settings = render_sidebar()

    # 메인 컨텐츠
    st.title("자동 자막 생성기")
    st.write("음성 또는 영상 파일을 업로드하여 자동으로 자막을 생성하세요.")

    # 고급 옵션
    with st.expander("옵션 설명", expanded=False):
        st.info("VAD(Voice Activity Detection)는 오디오에서 음성이 있는 부분만 감지하여 처리합니다. 중간 중간 오디오 공백이 있는 영상 및 음성 파일에서 효과적입니다.")
        st.warning("webrtcvad 모듈이 설치되지 않은 경우 VAD 기능이 비활성화됩니다.")

    # 자막 히스토리 표시
    render_subtitle_history()

    # 이전 자막 미리보기
    render_subtitle_preview()

    # 파일 업로드 및 자막 생성
    render_file_upload(settings)

    # 푸터
    render_footer()


if __name__ == "__main__":
    main()
