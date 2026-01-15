"""LLM 모델 관련 유틸리티"""

import streamlit as st
from openai import OpenAI


def get_openai_models(api_key):
    """OpenAI API에서 사용 가능한 모델 목록을 가져옵니다."""
    if not api_key:
        return []

    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()

        # 필터링: GPT 계열만, 오디오/임베딩/이미지 모델 제외
        filtered_models = []
        exclude_patterns = [
            'whisper', 'tts', 'audio', 'embedding', 'dall-e',
            'davinci', 'curie', 'babbage', 'ada', 'instruct',
            'realtime', 'moderation', 'search', 'image'
        ]
        # 오래된 날짜 패턴 (2023년 이전 버전)
        old_date_patterns = ['0301', '0314', '0613', '0125', '1106']

        for model in models.data:
            model_id = model.id.lower()

            # GPT 계열 모델만 포함
            if not (model_id.startswith('gpt-') or model_id.startswith('o1') or model_id.startswith('o3') or model_id.startswith('o4')):
                continue

            # 제외 패턴 체크
            if any(pattern in model_id for pattern in exclude_patterns):
                continue

            # 오래된 날짜 버전 제외
            if any(date in model_id for date in old_date_patterns):
                continue

            # 미리보기/프리뷰 버전 제외
            if 'preview' in model_id:
                continue

            filtered_models.append(model.id)

        # 정렬: 최신 모델이 위로
        def sort_key(model_name):
            # o1, o3, o4 계열을 앞에
            if model_name.startswith('o4'):
                return (0, model_name)
            elif model_name.startswith('o3'):
                return (1, model_name)
            elif model_name.startswith('o1'):
                return (2, model_name)
            elif 'gpt-4o' in model_name:
                return (3, model_name)
            elif 'gpt-4' in model_name:
                return (4, model_name)
            elif 'gpt-3.5' in model_name:
                return (5, model_name)
            else:
                return (6, model_name)

        filtered_models.sort(key=sort_key)
        return filtered_models

    except Exception as e:
        st.warning(f"OpenAI 모델 목록 가져오기 실패: {str(e)}")
        # 기본 모델 목록 반환
        return ["gpt-5-mini", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]


def get_anthropic_models(api_key):
    """Anthropic에서 사용 가능한 모델 목록을 반환합니다.
    Anthropic API는 모델 목록 엔드포인트가 없으므로 하드코딩합니다."""
    if not api_key:
        return []

    # Anthropic 최신 모델 목록 (텍스트 생성용만)
    models = [
        "claude-haiku-4-5",
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    return models
