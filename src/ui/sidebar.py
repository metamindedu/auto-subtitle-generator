"""사이드바 UI 모듈"""

import os
import torch
import streamlit as st

from ..config.api_keys import load_saved_api_keys, save_api_keys, delete_saved_api_keys
from ..utils.gpu_utils import display_gpu_info
from ..utils.llm_utils import get_openai_models, get_anthropic_models
from ..utils.vad_utils import is_vad_available


def render_sidebar():
    """사이드바 설정 UI 렌더링

    Returns:
        dict: 설정 값들을 담은 딕셔너리
    """
    settings = {}

    with st.sidebar:
        st.title("설정")

        # GPU 정보
        with st.expander("GPU 정보", expanded=False):
            display_gpu_info()

        # GPU 최적화 옵션
        if torch.cuda.is_available():
            with st.expander("GPU 최적화 옵션", expanded=False):
                st.info("최신 GPU는 Whisper 모델에는 높은 사용률이 필요하지 않을 수 있습니다.")
                use_half_precision = st.checkbox("Half Precision 사용 (FP16, 메모리 절약)", value=True)
                device_id = st.selectbox(
                    "GPU 장치 선택",
                    options=list(range(torch.cuda.device_count())),
                    format_func=lambda x: f"GPU {x}: {torch.cuda.get_device_name(x)}",
                    index=0
                )

                if use_half_precision:
                    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        # API 키 설정
        with st.expander("API 키 설정", expanded=False):
            openai_key = st.text_input(
                "OpenAI API 키",
                type="password",
                value=st.session_state.openai_api_key,
                key="openai_api_key_input"
            )
            anthropic_key = st.text_input(
                "Anthropic API 키",
                type="password",
                value=st.session_state.anthropic_api_key,
                key="anthropic_api_key_input"
            )

            # API 키 저장 체크박스
            save_keys = st.checkbox(
                "API 키 저장 (다음 실행 시에도 유지)",
                value=st.session_state.save_api_keys_enabled,
                help="체크하면 API 키가 로컬 파일에 저장됩니다. 저장된 키는 .gitignore에 의해 GitHub에 업로드되지 않습니다."
            )

            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
                st.session_state.openai_api_key = openai_key

            if anthropic_key:
                os.environ["ANTHROPIC_API_KEY"] = anthropic_key
                st.session_state.anthropic_api_key = anthropic_key

            # 저장 상태 변경 처리
            if save_keys != st.session_state.save_api_keys_enabled:
                st.session_state.save_api_keys_enabled = save_keys
                if save_keys:
                    # 체크박스 활성화: API 키 저장
                    if save_api_keys(openai_key, anthropic_key):
                        st.success("API 키가 저장되었습니다.")
                else:
                    # 체크박스 비활성화: 저장된 API 키 삭제
                    if delete_saved_api_keys():
                        st.info("저장된 API 키가 삭제되었습니다.")
            elif save_keys and (openai_key or anthropic_key):
                # 체크박스가 활성화된 상태에서 키가 변경되면 자동 저장
                save_api_keys(openai_key if openai_key else None, anthropic_key if anthropic_key else None)

        # 모델 설정
        settings['whisper_model'] = st.selectbox(
            "Whisper 모델 크기",
            options=["tiny", "base", "small", "medium", "large"],
            index=2
        )

        llm_provider_choice = st.radio(
            "LLM 교정 제공자",
            options=["사용안함", "OpenAI", "Anthropic"],
            index=0
        )

        # 선택된 제공자에 따라 모델 선택 UI 표시
        settings['llm_provider'] = None
        settings['llm_model'] = None

        if llm_provider_choice == "OpenAI":
            settings['llm_provider'] = "openai"
            openai_key = st.session_state.get('openai_api_key', '')

            if openai_key:
                with st.spinner("OpenAI 모델 목록 가져오는 중..."):
                    # 캐싱을 위한 세션 상태 사용
                    if 'openai_models_cache' not in st.session_state:
                        st.session_state.openai_models_cache = get_openai_models(openai_key)
                    models = st.session_state.openai_models_cache

                if models:
                    # gpt-5-mini를 기본값으로 선택
                    default_index = 0
                    if "gpt-5-mini" in models:
                        default_index = models.index("gpt-5-mini")

                    settings['llm_model'] = st.selectbox(
                        "OpenAI 모델 선택",
                        options=models,
                        index=default_index,
                        help="자막 교정에 사용할 OpenAI 모델을 선택하세요"
                    )
                else:
                    st.warning("사용 가능한 모델이 없습니다.")
                    settings['llm_model'] = "gpt-5-mini"

                # 모델 목록 새로고침 버튼
                if st.button("모델 목록 새로고침", key="refresh_openai_models"):
                    st.session_state.openai_models_cache = get_openai_models(openai_key)
                    st.rerun()
            else:
                st.warning("OpenAI API 키를 먼저 입력해주세요.")

        elif llm_provider_choice == "Anthropic":
            settings['llm_provider'] = "anthropic"
            anthropic_key = st.session_state.get('anthropic_api_key', '')

            if anthropic_key:
                models = get_anthropic_models(anthropic_key)

                if models:
                    settings['llm_model'] = st.selectbox(
                        "Anthropic 모델 선택",
                        options=models,
                        index=0 if models else None,
                        help="자막 교정에 사용할 Anthropic 모델을 선택하세요"
                    )
                else:
                    st.warning("사용 가능한 모델이 없습니다.")
                    settings['llm_model'] = "claude-haiku-4-5"
            else:
                st.warning("Anthropic API 키를 먼저 입력해주세요.")

        # VAD 설정
        settings['vad_enabled'] = st.checkbox("VAD(Voice Activity Detection) 사용", value=True,
                               help="음성이 있는 부분만 감지하여 처리합니다. 비활성화하면 전체 오디오를 한 번에 처리합니다.")

        if settings['vad_enabled'] and is_vad_available():
            settings['vad_aggressiveness'] = st.slider("VAD 감도", min_value=0, max_value=3, value=1,
                                        help="높을수록 더 엄격하게 음성을 감지합니다. 0: 매우 관대, 3: 매우 엄격")
        else:
            settings['vad_aggressiveness'] = 1

        # 자막 설정
        language = st.selectbox(
            "자막 언어",
            options=["자동 감지", "한국어", "영어", "일본어", "중국어"],
            index=0
        )

        settings['lang_code'] = None
        if language == "한국어":
            settings['lang_code'] = "ko"
        elif language == "영어":
            settings['lang_code'] = "en"
        elif language == "일본어":
            settings['lang_code'] = "ja"
        elif language == "중국어":
            settings['lang_code'] = "zh"

        max_chars = st.number_input("한 자막당 최대 글자 수", min_value=0, value=40)
        settings['max_chars'] = max_chars if max_chars > 0 else None

        min_chars = st.number_input("한 자막당 최소 글자 수", min_value=0, value=6)
        settings['min_chars'] = min_chars if min_chars > 0 else None

        max_duration = st.number_input("한 자막당 최대 시간(초)", min_value=0.0, value=10.0)
        settings['max_duration'] = max_duration if max_duration > 0 else None

        settings['context'] = st.text_area("영상 컨텍스트 (영상의 주제, 목적, 대상 청중 등)", height=100)
        if not settings['context']:
            settings['context'] = None

    return settings
