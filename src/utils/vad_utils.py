"""VAD (Voice Activity Detection) 유틸리티"""

import collections
import contextlib
import wave
import streamlit as st

# VAD 모듈 로드 시도
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False


class Frame:
    """VAD를 위한 프레임 클래스"""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """오디오를 프레임으로 분할"""
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """VAD를 사용하여 음성 구간 감지"""
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])

            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])

            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield (voiced_frames[0].timestamp,
                      voiced_frames[-1].timestamp + voiced_frames[-1].duration,
                      b''.join([f.bytes for f in voiced_frames]))
                ring_buffer.clear()
                voiced_frames = []

    if voiced_frames:
        yield (voiced_frames[0].timestamp,
               voiced_frames[-1].timestamp + voiced_frames[-1].duration,
               b''.join([f.bytes for f in voiced_frames]))


def process_with_vad(audio_path, aggressiveness=1):
    """VAD를 사용하여 음성 구간 처리"""
    if not VAD_AVAILABLE:
        # VAD를 사용할 수 없는 경우 전체 오디오를 하나의 세그먼트로 처리
        import soundfile as sf
        info = sf.info(audio_path)
        return [(0, info.duration)]

    with contextlib.closing(wave.open(audio_path, 'rb')) as wf:
        pcm_data = wf.readframes(wf.getnframes())
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()

    vad = webrtcvad.Vad(aggressiveness)
    frames = frame_generator(30, pcm_data, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 2000, vad, frames)

    voice_segments = []
    for start, end, _ in segments:
        voice_segments.append((start, end))

    return voice_segments


def is_vad_available():
    """VAD 모듈 사용 가능 여부 반환"""
    return VAD_AVAILABLE
