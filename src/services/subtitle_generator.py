"""자막 생성기 모듈"""

import os
import tempfile
import re
import json
import numpy as np
import pysrt
import streamlit as st
import whisper
from openai import OpenAI
import anthropic

from ..services.prompt_manager import PromptManager
from ..utils.vad_utils import process_with_vad


class SubtitleGenerator:
    """자막 생성기 클래스"""

    def __init__(self, model_size="small", llm_provider=None, llm_model=None):
        with st.spinner("Whisper 모델 로딩 중..."):
            self.model = whisper.load_model(model_size)
        st.success("모델 로딩 완료!")

        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.prompt_manager = PromptManager()

        self.llm_client = None
        if llm_provider == "openai":
            openai_api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get('openai_api_key')
            if openai_api_key:
                self.llm_client = OpenAI(api_key=openai_api_key)
            else:
                st.warning("OpenAI API 키가 필요합니다. 설정에서 입력해주세요.")
        elif llm_provider == "anthropic":
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or st.session_state.get('anthropic_api_key')
            if anthropic_api_key:
                self.llm_client = anthropic.Anthropic(api_key=anthropic_api_key)
            else:
                st.warning("Anthropic API 키가 필요합니다. 설정에서 입력해주세요.")

    def convert_to_wav(self, input_file):
        """업로드된 파일을 WAV 형식으로 변환"""
        try:
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
                temp_wav_path = temp_wav.name

            # 원본 파일을 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(input_file.name)[1]) as temp_input:
                temp_input.write(input_file.getbuffer())
                temp_input_path = temp_input.name

            # FFmpeg를 사용한 변환
            try:
                import ffmpeg

                # FFmpeg를 사용하여 변환
                (
                    ffmpeg
                    .input(temp_input_path)
                    .output(temp_wav_path, acodec='pcm_s16le', ar=16000, ac=1)
                    .run(quiet=True, overwrite_output=True)
                )

                # 파일 길이 확인
                probe = ffmpeg.probe(temp_wav_path)
                audio_info = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
                total_seconds = float(audio_info.get('duration', 0))

                return temp_wav_path, total_seconds, temp_input_path

            except ImportError:
                st.error("ffmpeg-python 패키지가 설치되지 않았습니다. 'pip install ffmpeg-python'으로 설치하세요.")
                raise
            except ffmpeg.Error as e:
                st.error(f"FFmpeg 변환 오류: {e.stderr.decode() if e.stderr else str(e)}")
                raise

        except Exception as e:
            st.error(f"오디오 변환 중 오류 발생: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

            # 임시 파일 정리
            if 'temp_input_path' in locals() and os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)

            return None, None, None

    def merge_short_subtitles(self, subtitles, min_chars):
        """짧은 자막들을 병합"""
        if not subtitles:
            return subtitles

        merged = []
        current = subtitles[0]

        for next_sub in subtitles[1:]:
            # 현재 자막이 최소 글자 수보다 적고, 다음 자막과의 시간 간격이 2초 이내인 경우
            if (len(current.text) < min_chars and
                (next_sub.start.hours * 3600 + next_sub.start.minutes * 60 + next_sub.start.seconds) -
                (current.end.hours * 3600 + current.end.minutes * 60 + current.end.seconds) <= 2):

                # 병합된 텍스트가 최대 글자 수를 초과하지 않는 경우에만 병합
                if len(current.text + " " + next_sub.text) <= 100:  # 기본 최대 글자 수 제한
                    current.text += " " + next_sub.text
                    current.end = next_sub.end
                else:
                    merged.append(current)
                    current = next_sub
            else:
                merged.append(current)
                current = next_sub

        merged.append(current)

        # 인덱스 재정렬
        for i, sub in enumerate(merged, 1):
            sub.index = i

        return merged

    def correct_subtitle_with_llm(self, subtitle_text, context=None, previous_subs=None, next_subs=None):
        """LLM을 사용하여 자막 텍스트를 교정"""
        if not self.llm_client:
            return subtitle_text

        try:
            # 원본 자막 로그 추가
            log_entry = f"원본 자막: {subtitle_text}"
            if 'correction_logs' not in st.session_state:
                st.session_state.correction_logs = []
            st.session_state.correction_logs.append(log_entry)

            if self.llm_provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model or "gpt-5-mini",
                    messages=[
                        {"role": "system", "content": self.prompt_manager.system_prompt},
                        {"role": "user", "content": self.prompt_manager.get_user_prompt(
                            context, subtitle_text, previous_subs, next_subs
                        )}
                    ],
                )
                corrected_text = response.choices[0].message.content.strip()

            elif self.llm_provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.llm_model or "claude-haiku-4-5",
                    max_tokens=1000,
                    system=self.prompt_manager.system_prompt,
                    messages=[{
                        "role": "user",
                        "content": self.prompt_manager.get_user_prompt(
                            context, subtitle_text, previous_subs, next_subs
                        )
                    }]
                )
                corrected_text = response.content[0].text.strip()

            # 교정된 자막 로그 추가
            log_entry = f"교정된 자막: {corrected_text}"
            st.session_state.correction_logs.append(log_entry)

            # 자동 스크롤을 위해 세션 상태 업데이트
            st.session_state.log_updated = True

            return corrected_text

        except Exception as e:
            error_msg = f"자막 교정 중 오류 발생: {str(e)}"
            st.session_state.correction_logs.append(error_msg)
            return subtitle_text

    def correct_subtitles_batch(self, segments, batch_size=10, context=None):
        """배치 단위로 자막을 교정

        Args:
            segments: [{"start": float, "end": float, "text": str}, ...] 형태의 리스트
            batch_size: 한 번에 처리할 자막 개수
            context: 영상 컨텍스트

        Returns:
            교정된 segments 리스트 (원본 구조 유지)
        """
        if not self.llm_client:
            return segments

        total_segments = len(segments)

        for batch_start in range(0, total_segments, batch_size):
            batch_end = min(batch_start + batch_size, total_segments)
            batch = segments[batch_start:batch_end]

            # 배치 프롬프트 생성 (타임코드 제외, 텍스트와 인덱스만 전달)
            subtitles_with_index = [
                {"index": i, "text": batch[i]["text"]}
                for i in range(len(batch))
            ]

            # 로그에 원본 자막 기록
            for item in subtitles_with_index:
                log_entry = f"원본 자막 [{batch_start + item['index'] + 1}]: {item['text']}"
                if 'correction_logs' not in st.session_state:
                    st.session_state.correction_logs = []
                st.session_state.correction_logs.append(log_entry)

            try:
                user_prompt = self.prompt_manager.get_batch_user_prompt(context, subtitles_with_index)

                if self.llm_provider == "openai":
                    response = self.llm_client.chat.completions.create(
                        model=self.llm_model or "gpt-5-mini",
                        messages=[
                            {"role": "system", "content": self.prompt_manager.system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                    )
                    response_text = response.choices[0].message.content.strip()

                elif self.llm_provider == "anthropic":
                    response = self.llm_client.messages.create(
                        model=self.llm_model or "claude-haiku-4-5",
                        max_tokens=4000,
                        system=self.prompt_manager.system_prompt,
                        messages=[{"role": "user", "content": user_prompt}]
                    )
                    response_text = response.content[0].text.strip()

                # JSON 파싱 (```json ... ``` 형식 처리)
                raw_response = response_text  # 디버깅용 원본 보존
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()

                try:
                    corrected_list = json.loads(response_text)
                except json.JSONDecodeError:
                    # 파싱 실패시 원본 응답 로그에 기록
                    error_msg = f"JSON 파싱 실패. LLM 원본 응답: {raw_response[:500]}"
                    st.session_state.correction_logs.append(error_msg)
                    raise

                # 응답 개수 검증
                if len(corrected_list) != len(batch):
                    error_msg = f"배치 교정 오류: 입력 {len(batch)}개, 출력 {len(corrected_list)}개 불일치"
                    st.session_state.correction_logs.append(error_msg)
                    continue

                # 교정 결과 적용 (index 매칭)
                for corrected_item in corrected_list:
                    idx = corrected_item.get("index")
                    # 여러 가능한 키 이름 시도 (LLM 응답 형식 변동 대응)
                    corrected_text = (
                        corrected_item.get("corrected") or
                        corrected_item.get("text") or
                        corrected_item.get("corrected_text") or
                        ""
                    )

                    if idx is not None and 0 <= idx < len(batch):
                        original_text = batch[idx]["text"]

                        # 교정된 텍스트가 비어있으면 원본 유지
                        if corrected_text and corrected_text.strip():
                            batch[idx]["text"] = corrected_text
                            final_text = corrected_text
                        else:
                            # 원본 유지하고 경고 로그 기록
                            final_text = original_text
                            warning_msg = f"경고: 자막 [{batch_start + idx + 1}] 교정 결과가 비어있어 원본 유지"
                            st.session_state.correction_logs.append(warning_msg)

                        # 로그에 교정된 자막 기록
                        log_entry = f"교정된 자막 [{batch_start + idx + 1}]: {final_text}"
                        st.session_state.correction_logs.append(log_entry)

            except json.JSONDecodeError as e:
                error_msg = f"배치 교정 JSON 파싱 오류: {str(e)}"
                st.session_state.correction_logs.append(error_msg)
            except Exception as e:
                error_msg = f"배치 교정 중 오류 발생: {str(e)}"
                st.session_state.correction_logs.append(error_msg)

        return segments

    def _update_correction_log_display(self, log_placeholder):
        """교정 로그 디스플레이 업데이트"""
        if 'correction_logs' not in st.session_state or not st.session_state.correction_logs:
            return

        # HTML 형식으로 로그 구성
        log_html = "<div class='log-container' id='log-container'>"
        reversed_logs = list(reversed(st.session_state.correction_logs))

        for i, log in enumerate(reversed_logs):
            if log.startswith("원본 자막"):
                log_html += f"<div class='original-subtitle'>{log}</div>"
            elif log.startswith("교정된 자막"):
                if i < len(st.session_state.correction_logs) - 1:
                    log_html += "<div class='log-divider'></div>"
                log_html += f"<div class='corrected-subtitle'>{log}</div>"
            elif "오류" in log:
                log_html += f"<div class='error-message'>{log}</div>"

        log_html += "</div>"

        # 로그 표시
        log_placeholder.markdown(log_html, unsafe_allow_html=True)

    def generate_subtitles(self, audio_file, progress_bar, status_text, language=None, max_chars=None, min_chars=None, max_duration=None, context=None, vad_enabled=True, vad_aggressiveness=1):
        """자막 생성 함수"""
        temp_files = []

        # 로그 컨테이너 초기화
        st.session_state.correction_logs = []
        process_container = st.container()
        status_container = process_container.container()
        log_container = process_container.container()
        # 교정 로그 스타일 정의
        log_container.markdown("""
        <style>
        .log-container {
            height: 250px;
            overflow-y: auto;
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            font-family: monospace;
            border: 1px solid #ddd;
        }
        .original-subtitle {
            color: #555;
            margin-bottom: 4px;
        }
        .corrected-subtitle {
            color: #0066cc;
            margin-bottom: 12px;
            font-weight: bold;
        }
        .error-message {
            color: #cc0000;
            font-weight: bold;
        }
        .log-divider {
            border-bottom: 1px dashed #ccc;
            margin: 8px 0;
        }
        </style>
        """, unsafe_allow_html=True)

        # 로그 표시 영역
        log_heading = log_container.empty()
        log_placeholder = log_container.empty()

        # 로그 제목 설정
        if self.llm_client:
            log_heading.subheader("실시간 자막 교정 로그")

        try:
            # WAV 파일 생성
            wav_path, total_seconds, temp_input_path = self.convert_to_wav(audio_file)
            if not wav_path:
                return None

            temp_files.extend([wav_path, temp_input_path])
            status_text.text("WAV 파일 생성 완료")
            progress_bar.progress(10)

            # VAD를 사용하여 음성 구간 감지
            if vad_enabled:
                status_text.text("음성 구간 감지 중...")
                voice_segments = process_with_vad(wav_path, vad_aggressiveness)
                status_text.text(f"감지된 음성 구간: {len(voice_segments)}개")
            else:
                # VAD를 사용하지 않는 경우 전체 오디오를 하나의 세그먼트로 처리
                import soundfile as sf
                info = sf.info(wav_path)
                voice_segments = [(0, info.duration)]
                status_text.text("VAD 비활성화: 전체 오디오를 한 번에 처리합니다")

            progress_bar.progress(20)

            # 전체 자막 정보 저장용
            all_segments = []

            # 오디오 파일 로드
            import soundfile as sf
            audio, sample_rate = sf.read(wav_path)
            audio = audio.astype(np.float32)

            # 음성 구간 처리
            status_text.text("음성 인식 시작...")
            total_segments = len(voice_segments)

            for i, (start, end) in enumerate(voice_segments):
                status_text.text(f"음성 구간 {i+1}/{total_segments} 처리 중...")

                # 진행률 업데이트
                segment_progress = 20 + (i / total_segments * 40)
                progress_bar.progress(int(segment_progress))

                # 현재 구간의 오디오 추출
                start_sample = int(start * sample_rate)
                end_sample = min(int(end * sample_rate), len(audio))
                segment_audio = audio[start_sample:end_sample]

                # Whisper로 음성 인식
                transcribe_options = {}
                if language:
                    transcribe_options["language"] = language

                result = self.model.transcribe(segment_audio, **transcribe_options)

                # 세그먼트 시간 조정 및 정보 저장
                for segment in result["segments"]:
                    adj_segment = {
                        "start": start + segment["start"],
                        "end": start + segment["end"],
                        "text": segment["text"].strip(),
                        "processed": False  # 이 세그먼트가 처리되었는지 표시
                    }
                    all_segments.append(adj_segment)

            status_text.text("음성 인식 완료!")
            progress_bar.progress(60)

            # LLM 교정 처리 (배치 방식)
            if self.llm_client:
                status_text.text("LLM 자막 교정 시작 (배치 처리)...")

                batch_size = 10  # 한 번에 처리할 자막 개수
                total_segments = len(all_segments)
                total_batches = (total_segments + batch_size - 1) // batch_size

                for batch_idx in range(total_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, total_segments)

                    # 진행률 업데이트
                    segment_progress = 60 + ((batch_idx / total_batches) * 30)
                    progress_bar.progress(int(segment_progress))

                    status_text.text(f"자막 교정 중... (배치 {batch_idx+1}/{total_batches}, 자막 {batch_start+1}-{batch_end}/{total_segments})")

                    # 배치 교정 수행
                    batch_segments = all_segments[batch_start:batch_end]
                    self.correct_subtitles_batch(batch_segments, batch_size=len(batch_segments), context=context)

                    # 로그 업데이트 및 화면 갱신
                    self._update_correction_log_display(log_placeholder)

            # 자막 파일 생성
            status_text.text("자막 파일 생성 중...")
            subs = pysrt.SubRipFile()
            subtitle_index = 1

            # 텍스트 분할 및 자막 생성
            for segment in all_segments:
                if not segment["text"]:  # 빈 텍스트는 건너뛰기
                    continue

                text = segment["text"]
                start_time = segment["start"]
                end_time = segment["end"]
                duration = end_time - start_time

                # 최대 시간 길이 체크
                if max_duration and duration > max_duration:
                    # 시간 간격으로 분할
                    num_splits = int(np.ceil(duration / max_duration))
                    sub_duration = duration / num_splits

                    # 텍스트를 글자 수에 비례하여 공정하게 분할
                    splits = []

                    # 텍스트를 문장 단위로 분할 (더 자연스러운 분할 지점)
                    sentence_breaks = re.split(r'([.!?] )', text)
                    sentences = []

                    # 문장 재구성 (구분자 포함)
                    i = 0
                    while i < len(sentence_breaks):
                        if i + 1 < len(sentence_breaks) and re.match(r'[.!?] ', sentence_breaks[i+1]):
                            sentences.append(sentence_breaks[i] + sentence_breaks[i+1])
                            i += 2
                        else:
                            sentences.append(sentence_breaks[i])
                            i += 1

                    # 빈 문장 제거
                    sentences = [s for s in sentences if s.strip()]

                    if not sentences:  # 문장이 없으면 단어 단위로 분할
                        words = text.split()
                        chars_per_split = len(text) / num_splits
                        current_split = []
                        current_length = 0

                        for word in words:
                            if current_length + len(word) + (1 if current_length > 0 else 0) <= chars_per_split:
                                current_split.append(word)
                                current_length += len(word) + (1 if current_length > 0 else 0)
                            else:
                                if current_split:  # 현재 분할을 추가
                                    splits.append(' '.join(current_split))
                                current_split = [word]
                                current_length = len(word)

                        if current_split:  # 마지막 분할 추가
                            splits.append(' '.join(current_split))
                    else:
                        # 문장을 적절히 그룹화하여 분할
                        current_split = []
                        current_length = 0
                        target_length = len(text) / num_splits

                        for sentence in sentences:
                            if current_length + len(sentence) <= target_length * 1.3:  # 30% 여유 허용
                                current_split.append(sentence)
                                current_length += len(sentence)
                            else:
                                if current_split:
                                    splits.append(''.join(current_split).strip())
                                current_split = [sentence]
                                current_length = len(sentence)

                        if current_split:
                            splits.append(''.join(current_split).strip())

                    # 필요한 경우 분할 수 맞추기
                    while len(splits) < num_splits:
                        # 가장 긴 분할을 찾아 분할
                        longest_idx = max(range(len(splits)), key=lambda i: len(splits[i]))
                        longest_split = splits[longest_idx]

                        if len(longest_split) < 10:  # 너무 짧으면 분할하지 않음
                            break

                        mid_point = len(longest_split) // 2
                        # 공백을 찾아 분할점 조정
                        while mid_point > 0 and mid_point < len(longest_split) - 1:
                            if longest_split[mid_point] == ' ':
                                break
                            mid_point += 1

                        if mid_point == 0 or mid_point >= len(longest_split) - 1:
                            # 적절한 분할점을 찾지 못하면 그냥 중간에서 자름
                            mid_point = len(longest_split) // 2

                        first_half = longest_split[:mid_point].strip()
                        second_half = longest_split[mid_point:].strip()

                        splits[longest_idx] = first_half
                        splits.insert(longest_idx + 1, second_half)

                    # 시간 분배
                    current_time = start_time
                    for split_text in splits:
                        split_ratio = len(split_text) / sum(len(s) for s in splits)
                        split_duration = duration * split_ratio
                        split_end = current_time + split_duration

                        if max_chars and len(split_text) > max_chars:
                            # 글자 수가 너무 많으면 추가 분할 (단어 경계 유지)
                            words = split_text.split()
                            current_chunk = ""
                            chunks = []

                            for word in words:
                                test_chunk = (current_chunk + " " + word).strip()
                                if len(test_chunk) <= max_chars:
                                    current_chunk = test_chunk
                                else:
                                    if current_chunk:
                                        chunks.append(current_chunk)
                                    current_chunk = word

                            if current_chunk:
                                chunks.append(current_chunk)

                            # 분할된 청크에 시간 배분
                            chunk_time = current_time
                            chunk_duration = split_duration / len(chunks)
                            for chunk in chunks:
                                chunk_end = chunk_time + chunk_duration

                                # 자막 아이템 생성 및 추가
                                hours = int(chunk_time) // 3600
                                minutes = (int(chunk_time) % 3600) // 60
                                seconds = int(chunk_time) % 60
                                milliseconds = int((chunk_time % 1) * 1000)
                                start_time_item = pysrt.SubRipTime(hours=hours, minutes=minutes,
                                                        seconds=seconds, milliseconds=milliseconds)

                                hours = int(chunk_end) // 3600
                                minutes = (int(chunk_end) % 3600) // 60
                                seconds = int(chunk_end) % 60
                                milliseconds = int((chunk_end % 1) * 1000)
                                end_time_item = pysrt.SubRipTime(hours=hours, minutes=minutes,
                                                    seconds=seconds, milliseconds=milliseconds)

                                sub = pysrt.SubRipItem(
                                    index=subtitle_index,
                                    start=start_time_item,
                                    end=end_time_item,
                                    text=chunk
                                )
                                subs.append(sub)
                                subtitle_index += 1

                                chunk_time = chunk_end
                        else:
                            # 최대 글자 수 이내인 경우 단일 자막으로 추가
                            hours = int(current_time) // 3600
                            minutes = (int(current_time) % 3600) // 60
                            seconds = int(current_time) % 60
                            milliseconds = int((current_time % 1) * 1000)
                            start_time_item = pysrt.SubRipTime(hours=hours, minutes=minutes,
                                                    seconds=seconds, milliseconds=milliseconds)

                            hours = int(split_end) // 3600
                            minutes = (int(split_end) % 3600) // 60
                            seconds = int(split_end) % 60
                            milliseconds = int((split_end % 1) * 1000)
                            end_time_item = pysrt.SubRipTime(hours=hours, minutes=minutes,
                                                seconds=seconds, milliseconds=milliseconds)

                            sub = pysrt.SubRipItem(
                                index=subtitle_index,
                                start=start_time_item,
                                end=end_time_item,
                                text=split_text
                            )
                            subs.append(sub)
                            subtitle_index += 1

                        current_time = split_end
                else:
                    # 최대 시간 이내인 경우 최대 글자 수에 따라 처리
                    if max_chars and len(text) > max_chars:
                        # 단어 단위로 분할
                        words = text.split()
                        current_text = ""
                        sub_splits = []

                        for word in words:
                            test_text = (current_text + " " + word).strip()
                            if len(test_text) <= max_chars:
                                current_text = test_text
                            else:
                                if current_text:
                                    sub_splits.append(current_text)
                                current_text = word

                        if current_text:  # 마지막 부분 추가
                            sub_splits.append(current_text)

                        # 시간을 텍스트 길이에 비례하여 분배
                        sub_duration = end_time - start_time
                        total_chars = sum(len(s) for s in sub_splits)
                        current_time = start_time

                        for sub_text in sub_splits:
                            ratio = len(sub_text) / total_chars
                            split_duration = sub_duration * ratio
                            split_end = current_time + split_duration

                            hours = int(current_time) // 3600
                            minutes = (int(current_time) % 3600) // 60
                            seconds = int(current_time) % 60
                            milliseconds = int((current_time % 1) * 1000)
                            start_time_item = pysrt.SubRipTime(hours=hours, minutes=minutes,
                                                    seconds=seconds, milliseconds=milliseconds)

                            hours = int(split_end) // 3600
                            minutes = (int(split_end) % 3600) // 60
                            seconds = int(split_end) % 60
                            milliseconds = int((split_end % 1) * 1000)
                            end_time_item = pysrt.SubRipTime(hours=hours, minutes=minutes,
                                                seconds=seconds, milliseconds=milliseconds)

                            sub = pysrt.SubRipItem(
                                index=subtitle_index,
                                start=start_time_item,
                                end=end_time_item,
                                text=sub_text
                            )
                            subs.append(sub)
                            subtitle_index += 1

                            current_time = split_end
                    else:
                        # 그대로 추가
                        hours = int(start_time) // 3600
                        minutes = (int(start_time) % 3600) // 60
                        seconds = int(start_time) % 60
                        milliseconds = int((start_time % 1) * 1000)
                        start_time_item = pysrt.SubRipTime(hours=hours, minutes=minutes,
                                                seconds=seconds, milliseconds=milliseconds)

                        hours = int(end_time) // 3600
                        minutes = (int(end_time) % 3600) // 60
                        seconds = int(end_time) % 60
                        milliseconds = int((end_time % 1) * 1000)
                        end_time_item = pysrt.SubRipTime(hours=hours, minutes=minutes,
                                            seconds=seconds, milliseconds=milliseconds)

                        sub = pysrt.SubRipItem(
                            index=subtitle_index,
                            start=start_time_item,
                            end=end_time_item,
                            text=text
                        )
                        subs.append(sub)
                        subtitle_index += 1

            # 최소 글자 수 제한이 설정된 경우 짧은 자막 병합
            if min_chars:
                status_text.text("짧은 자막 병합 중...")
                merged_subs = self.merge_short_subtitles(subs, min_chars)
                subs = pysrt.SubRipFile()
                for sub in merged_subs:
                    subs.append(sub)

            # 임시 SRT 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix='.srt') as temp_srt:
                temp_srt_path = temp_srt.name
                temp_files.append(temp_srt_path)

            subs.save(temp_srt_path, encoding='utf-8')
            status_text.text("자막 파일 생성 완료!")
            progress_bar.progress(100)

            # SRT 파일 내용 읽기
            with open(temp_srt_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()

            return srt_content

        except Exception as e:
            st.error(f"자막 생성 중 오류 발생: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None

        finally:
            # 임시 파일 정리
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    def convert_srt_to_vtt(self, srt_content):
        """SRT 형식의 자막을 VTT 형식으로 변환합니다."""
        # VTT 헤더 추가
        vtt_content = "WEBVTT\n\n"

        # SRT 블록 단위로 분할
        srt_blocks = srt_content.strip().split('\n\n')

        for block in srt_blocks:
            lines = block.split('\n')

            # 각 블록은 최소 3줄 이상이어야 함 (인덱스, 시간, 텍스트)
            if len(lines) >= 3:
                # 인덱스 라인은 건너뛰기

                # 시간 포맷 변환 (00:00:00,000 --> 00:00:00.000)
                time_line = lines[1].replace(',', '.')

                # 텍스트 라인 유지
                text_lines = lines[2:]

                # VTT 블록 생성
                vtt_block = time_line + '\n' + '\n'.join(text_lines)
                vtt_content += vtt_block + '\n\n'

        return vtt_content
