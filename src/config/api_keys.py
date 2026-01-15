"""API 키 관리 모듈"""

import os
import json
import streamlit as st

# API 키 저장 파일 경로
API_KEYS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".api_keys.json")


def load_saved_api_keys():
    """저장된 API 키를 파일에서 로드"""
    try:
        if os.path.exists(API_KEYS_FILE):
            with open(API_KEYS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"API 키 로드 중 오류: {str(e)}")
    return {}


def save_api_keys(openai_key=None, anthropic_key=None):
    """API 키를 파일에 저장"""
    try:
        keys = load_saved_api_keys()
        if openai_key is not None:
            keys['openai_api_key'] = openai_key
        if anthropic_key is not None:
            keys['anthropic_api_key'] = anthropic_key
        with open(API_KEYS_FILE, 'w', encoding='utf-8') as f:
            json.dump(keys, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"API 키 저장 중 오류: {str(e)}")
        return False


def delete_saved_api_keys():
    """저장된 API 키 파일 삭제"""
    try:
        if os.path.exists(API_KEYS_FILE):
            os.remove(API_KEYS_FILE)
        return True
    except Exception as e:
        st.error(f"API 키 삭제 중 오류: {str(e)}")
        return False
