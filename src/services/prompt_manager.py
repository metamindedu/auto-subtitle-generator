"""프롬프트 관리 모듈"""

import os
import streamlit as st


class PromptManager:
    """LLM 프롬프트 관리 클래스"""

    def __init__(self, prompts_dir="prompts"):
        self.prompts_dir = prompts_dir
        self.system_prompt = self._load_prompt("system_prompt.md")
        self.user_prompt_template = self._load_prompt("user_prompt.md")
        self.user_prompt_batch_template = self._load_prompt("user_prompt_batch.md")

    def _load_prompt(self, filename):
        """마크다운 파일에서 프롬프트 로드"""
        try:
            prompt_path = os.path.join(self.prompts_dir, filename)
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            st.error(f"프롬프트 파일 로드 중 오류 발생 ({filename}): {str(e)}")
            return ""

    def get_user_prompt(self, context, current_sub, previous_subs, next_subs):
        """사용자 프롬프트 생성"""
        return self.user_prompt_template.format(
            context=context or '없음',
            previous_subs='\n'.join(previous_subs) if previous_subs else '없음',
            current_sub=current_sub,
            next_subs='\n'.join(next_subs) if next_subs else '없음'
        )

    def get_batch_user_prompt(self, context, subtitles_with_index):
        """배치 처리용 사용자 프롬프트 생성

        Args:
            context: 영상 컨텍스트
            subtitles_with_index: [{"index": 0, "text": "자막1"}, ...] 형태의 리스트
        """
        subtitle_list = "\n".join([
            f'{{"index": {item["index"]}, "text": "{item["text"]}"}}'
            for item in subtitles_with_index
        ])
        return self.user_prompt_batch_template.format(
            context=context or '없음',
            subtitle_list=subtitle_list
        )
