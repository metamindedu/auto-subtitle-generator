"""메인 컨텐츠 UI 모듈"""

import time
import streamlit as st

from ..services.subtitle_generator import SubtitleGenerator


def save_to_subtitle_history(filename, srt_content, vtt_content=None, correction_logs=None):
    """자막과 교정 로그를 히스토리에 저장하고 생성된 ID를 반환하는 함수"""
    if not filename or not srt_content:
        return None

    # 고유 식별자 생성 (파일명 + 타임스탬프)
    current_time = time.time()
    unique_id = f"{filename}_{current_time}"

    # 중복 확인
    exists = False
    existing_id = None
    for item in st.session_state.subtitle_history:
        if item['filename'] == filename and item['content'] == srt_content:
            exists = True
            existing_id = item.get('id')
            break

    # 중복이 아닌 경우에만 저장
    if not exists:
        # 자막 히스토리에 저장
        st.session_state.subtitle_history.append({
            'id': unique_id,
            'filename': filename,
            'content': srt_content,
            'vtt_content': vtt_content,
            'timestamp': current_time
        })

        # 교정 로그가 있으면 로그 히스토리에 저장
        if correction_logs:
            st.session_state.correction_logs_history[unique_id] = correction_logs.copy()

        # 목록이 너무 길어지는 것을 방지 (최대 10개 저장)
        if len(st.session_state.subtitle_history) > 10:
            # 가장 오래된 항목 제거
            oldest_item = st.session_state.subtitle_history.pop(0)
            # 관련 로그도 제거
            if oldest_item.get('id') in st.session_state.correction_logs_history:
                del st.session_state.correction_logs_history[oldest_item['id']]

        return unique_id
    else:
        # 이미 존재하는 경우 기존 ID 반환
        return existing_id


def render_subtitle_history():
    """자막 히스토리 UI 렌더링"""
    if 'subtitle_history' in st.session_state and len(st.session_state.subtitle_history) > 0:
        with st.expander("이전 자막 히스토리", expanded=False):
            for i, item in enumerate(reversed(st.session_state.subtitle_history)):
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

                with col1:
                    st.write(f"{item['filename']} ({time.strftime('%Y-%m-%d %H:%M', time.localtime(item['timestamp']))})")

                with col2:
                    st.download_button(
                        label="SRT",
                        data=item['content'],
                        file_name=f"{item['filename'].split('.')[0]}.srt",
                        mime="text/plain",
                        key=f"srt_download_{i}",
                        use_container_width=True
                    )

                with col3:
                    if item.get('vtt_content'):
                        st.download_button(
                            label="VTT",
                            data=item['vtt_content'],
                            file_name=f"{item['filename'].split('.')[0]}.vtt",
                            mime="text/plain",
                            key=f"vtt_download_{i}",
                            use_container_width=True
                        )
                    else:
                        st.write("VTT 없음")

                with col4:
                    if st.button("로드", key=f"load_history_{i}", use_container_width=True):
                        # 현재 표시 중인 자막이 있으면 히스토리에 저장 (히스토리 로드 전에)
                        if st.session_state.last_srt_content is not None and st.session_state.last_filename is not None:
                            save_to_subtitle_history(
                                st.session_state.last_filename,
                                st.session_state.last_srt_content,
                                st.session_state.get('last_vtt_content'),
                                st.session_state.get('correction_logs', [])
                            )

                        # 히스토리에서 선택한 자막 로드
                        st.session_state.last_srt_content = item['content']
                        st.session_state.last_filename = item['filename']
                        if item.get('vtt_content'):
                            st.session_state.last_vtt_content = item['vtt_content']

                        # 자막 ID 저장
                        st.session_state.current_subtitle_id = item.get('id')

                        # 관련 교정 로그가 있으면 로드
                        if item.get('id') in st.session_state.correction_logs_history:
                            st.session_state.correction_logs = st.session_state.correction_logs_history[item['id']].copy()
                        else:
                            # 로그가 없는 경우 빈 리스트로 초기화
                            st.session_state.correction_logs = []

                        st.rerun()

                st.markdown("---")


def render_subtitle_preview():
    """이전에 생성된 자막 미리보기 UI 렌더링"""
    if 'last_srt_content' not in st.session_state or not st.session_state.last_srt_content:
        return

    with st.container():
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

        # 파일명 확인
        if 'last_filename' in st.session_state and st.session_state.last_filename:
            video_title = st.session_state.last_filename
            file_name_base = video_title.split('.')[0]
        else:
            video_title = "Unknown"
            file_name_base = "subtitle"

        # 영상 제목 표시
        with col1:
            st.info(f"이전에 생성된 자막: {video_title}")

        # SRT 다운로드 버튼
        with col2:
            st.download_button(
                label="SRT 다운로드",
                data=st.session_state.last_srt_content,
                file_name=f"{file_name_base}.srt",
                mime="text/plain",
                use_container_width=True
            )

        # VTT 다운로드 버튼
        with col3:
            # SRT를 VTT로 변환
            if 'last_vtt_content' not in st.session_state or not st.session_state.last_vtt_content:
                # SubtitleGenerator 객체가 있으면 그것을 사용, 없으면 임시로 생성
                if 'subtitle_generator' in st.session_state and st.session_state.subtitle_generator:
                    generator = st.session_state.subtitle_generator
                else:
                    generator = SubtitleGenerator(model_size="small")

                st.session_state.last_vtt_content = generator.convert_srt_to_vtt(st.session_state.last_srt_content)

            st.download_button(
                label="VTT 다운로드",
                data=st.session_state.last_vtt_content,
                file_name=f"{file_name_base}.vtt",
                mime="text/plain",
                use_container_width=True
            )

        # 미리보기 버튼
        with col4:
            show_preview = st.button("미리보기", key="show_preview_button", use_container_width=True)

        # 미리보기 버튼이 클릭되면 자막 내용과 테이블 표시
        if show_preview:
            st.session_state.show_last_preview = True

            # 저장된 현재 자막 ID가 있으면 사용
            if 'current_subtitle_id' in st.session_state and st.session_state.current_subtitle_id:
                current_id = st.session_state.current_subtitle_id
                if current_id in st.session_state.correction_logs_history:
                    st.session_state.correction_logs = st.session_state.correction_logs_history[current_id].copy()
            else:
                # 현재 자막의 ID 찾기
                current_id = None
                for item in st.session_state.subtitle_history:
                    if (item['filename'] == st.session_state.last_filename and
                        item['content'] == st.session_state.last_srt_content):
                        current_id = item.get('id')
                        # ID를 찾았으면 해당 로그 불러오기
                        if current_id in st.session_state.correction_logs_history:
                            st.session_state.correction_logs = st.session_state.correction_logs_history[current_id].copy()
                            # 찾은 ID 저장
                            st.session_state.current_subtitle_id = current_id
                        break

        # 자막 미리보기 및 교정 로그를 하나의 화면에 표시
        if st.session_state.get('show_last_preview', False):
            _render_preview_content()


def _render_preview_content():
    """미리보기 컨텐츠 렌더링 (내부 함수)"""
    # 미리보기 닫기 버튼
    if st.button("미리보기 닫기", key="hide_preview_button1"):
        st.session_state.show_last_preview = False
        st.rerun()  # 페이지 새로고침

    # 교정 로그가 있는 경우 표시
    if st.session_state.get('correction_logs') and len(st.session_state.correction_logs) > 0:
        st.subheader("자막 교정 로그")

        # 교정 로그 표시 스타일
        st.markdown("""
        <style>
        .log-container {
            height: 400px;
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

        # 로그 표시
        log_html = "<div class='log-container' id='log-container'>"
        for i, log in enumerate(st.session_state.correction_logs):
            if log.startswith("원본 자막"):
                log_html += f"<div class='original-subtitle'>{log}</div>"
            elif log.startswith("교정된 자막"):
                log_html += f"<div class='corrected-subtitle'>{log}</div>"
                if i < len(st.session_state.correction_logs) - 1:
                    log_html += "<div class='log-divider'></div>"
            elif "오류" in log:
                log_html += f"<div class='error-message'>{log}</div>"
        log_html += "</div>"

        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.info("이 자막에 대한 교정 로그가 없습니다.")

    with st.expander("자막 내용", expanded=True):
        st.text_area("SRT 자막", st.session_state.last_srt_content, height=200)

    # 자막 미리보기 테이블
    st.subheader("자막 미리보기")
    srt_lines = st.session_state.last_srt_content.strip().split('\n\n')
    preview_data = []

    for block in srt_lines:
        lines = block.split('\n')
        if len(lines) >= 3:
            try:
                index = int(lines[0])
                time_info = lines[1]
                text = ' '.join(lines[2:])
                preview_data.append({"번호": index, "시간": time_info, "자막": text})
            except:
                pass

    if preview_data:
        st.dataframe(preview_data, use_container_width=True)

    # 미리보기 닫기 버튼
    if st.button("미리보기 닫기", key="hide_preview_button2"):
        st.session_state.show_last_preview = False
        st.rerun()  # 페이지 새로고침


def render_file_upload(settings):
    """파일 업로드 및 자막 생성 UI 렌더링

    Args:
        settings: 사이드바에서 받은 설정 딕셔너리
    """
    # 파일 업로드
    uploaded_file = st.file_uploader("음성 또는 영상 파일 업로드", type=["mp3", "wav", "mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("자막 생성 시작", type="primary"):
            # 로그 초기화
            st.session_state.correction_logs = []

            # 프로그레스 바와 상태 텍스트
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Whisper 모델 로딩 중...")

            # 자막 생성기 초기화
            generator = SubtitleGenerator(
                model_size=settings['whisper_model'],
                llm_provider=settings['llm_provider'],
                llm_model=settings['llm_model']
            )

            progress_bar.progress(10)
            status_text.text("자막 생성 중...")

            # 자막 생성
            srt_content = generator.generate_subtitles(
                audio_file=uploaded_file,
                progress_bar=progress_bar,
                status_text=status_text,
                language=settings['lang_code'],
                max_chars=settings['max_chars'],
                min_chars=settings['min_chars'],
                max_duration=settings['max_duration'],
                context=settings['context'],
                vad_enabled=settings['vad_enabled'],
                vad_aggressiveness=settings['vad_aggressiveness']
            )

            if srt_content:
                # 이전 자막을 히스토리에 저장 (기존 자막이 있는 경우)
                if st.session_state.last_srt_content is not None and st.session_state.last_filename is not None:
                    save_to_subtitle_history(
                        st.session_state.last_filename,
                        st.session_state.last_srt_content,
                        st.session_state.get('last_vtt_content'),
                        st.session_state.get('correction_logs', [])
                    )

                # 세션 상태에 자막 내용 저장
                st.session_state.last_srt_content = srt_content
                st.session_state.last_filename = uploaded_file.name
                st.session_state.last_vtt_content = generator.convert_srt_to_vtt(srt_content)

                # 새 자막을 히스토리에 저장하고 ID 바로 받기
                if st.session_state.get('correction_logs'):
                    new_subtitle_id = save_to_subtitle_history(
                        uploaded_file.name,
                        srt_content,
                        st.session_state.last_vtt_content,
                        st.session_state.correction_logs
                    )

                    # ID를 세션 상태에 저장하여 미리보기 시 사용
                    if new_subtitle_id:
                        st.session_state.current_subtitle_id = new_subtitle_id

                # 자막과 교정 로그를 연결하여 저장
                if len(st.session_state.get('correction_logs', [])) > 0:
                    # 방금 생성한 자막 찾기
                    for item in reversed(st.session_state.subtitle_history):
                        if (item['filename'] == uploaded_file.name and
                            item['content'] == srt_content):
                            # 교정 로그 저장
                            st.session_state.correction_logs_history[item['id']] = st.session_state.correction_logs.copy()
                            break

                # 자막 생성기 저장 (나중에 VTT 변환 등에 사용)
                if 'subtitle_generator' not in st.session_state:
                    st.session_state.subtitle_generator = generator

                # 자막 표시
                st.subheader("생성된 자막")
                st.text_area("SRT 자막", srt_content, height=300)

                # 다운로드 버튼들
                file_name_base = uploaded_file.name.split('.')[0]

                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.write("")  # 빈 공간

                with col2:
                    st.download_button(
                        label="SRT 파일 다운로드",
                        data=srt_content,
                        file_name=f"{file_name_base}.srt",
                        mime="text/plain",
                        use_container_width=True
                    )

                with col3:
                    st.download_button(
                        label="VTT 파일 다운로드",
                        data=st.session_state.last_vtt_content,
                        file_name=f"{file_name_base}.vtt",
                        mime="text/plain",
                        use_container_width=True
                    )

                # 미리보기 탭 추가
                st.subheader("자막 미리보기")

                # SRT 파싱
                srt_lines = srt_content.strip().split('\n\n')
                preview_data = []

                for block in srt_lines:
                    lines = block.split('\n')
                    if len(lines) >= 3:
                        try:
                            index = int(lines[0])
                            time_info = lines[1]
                            text = ' '.join(lines[2:])
                            preview_data.append({"번호": index, "시간": time_info, "자막": text})
                        except:
                            pass

                if preview_data:
                    st.dataframe(preview_data, use_container_width=True)


def render_footer():
    """푸터 UI 렌더링"""
    # 푸터 구분선
    st.markdown("---")

    # 푸터 컨테이너 생성
    footer = st.container()

    with footer:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                """
                <div style="text-align: center; padding: 10px;">
                    <p style="font-size: 0.9em; color: #666;">
                        이 프로그램은 <a href="https://metamind.kr" target="_blank" style="color: #4B9CFF; text-decoration: none;">메타마인드</a>가 제작하였으며, 자유로운 수정 및 공유가 가능합니다.
                    </p>
                    <p style="font-size: 0.8em; color: #888;">© 2025 메타마인드</p>
                </div>
                """,
                unsafe_allow_html=True
            )
