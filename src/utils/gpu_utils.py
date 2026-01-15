"""GPU 관련 유틸리티"""

import os
import torch
import streamlit as st


def check_gpu_status():
    """GPU 감지 및 사용 상태를 확인하는 함수"""
    gpu_info = {
        "is_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None,
        "device_name": None,
        "memory_allocated": None,
        "memory_reserved": None,
        "memory_total": None
    }

    if gpu_info["is_available"]:
        current_device = torch.cuda.current_device()
        gpu_info["current_device"] = current_device
        gpu_info["device_name"] = torch.cuda.get_device_name(current_device)

        # 단위 변환 함수 (바이트 -> GB)
        def bytes_to_gb(bytes_value):
            return round(bytes_value / (1024**3), 2)

        try:
            gpu_info["memory_allocated"] = bytes_to_gb(torch.cuda.memory_allocated(current_device))
            gpu_info["memory_reserved"] = bytes_to_gb(torch.cuda.memory_reserved(current_device))

            # 전체 VRAM 용량 확인 (Windows 전용)
            if os.name == 'nt':
                try:
                    # nvidia-smi 명령어 실행
                    import subprocess
                    result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                                               universal_newlines=True)
                    memory_total = int(result.strip())
                    gpu_info["memory_total"] = memory_total / 1024  # MB -> GB
                except:
                    gpu_info["memory_total"] = "확인 불가"
            else:
                gpu_info["memory_total"] = "확인 불가"
        except:
            # 메모리 정보를 가져올 수 없는 경우
            gpu_info["memory_allocated"] = "확인 불가"
            gpu_info["memory_reserved"] = "확인 불가"
            gpu_info["memory_total"] = "확인 불가"

    return gpu_info


def display_gpu_info():
    """GPU 정보를 Streamlit UI에 표시하는 함수"""
    gpu_info = check_gpu_status()

    # GPU 사용 가능 여부에 따라 다른 색상 및 메시지 표시
    if gpu_info["is_available"]:
        st.success("GPU 감지됨!")

        # GPU 정보 표시
        col1, col2 = st.columns(2)

        with col1:
            st.metric("감지된 GPU 수", gpu_info["device_count"])
            st.write(f"**모델**: {gpu_info['device_name']}")

        with col2:
            if isinstance(gpu_info["memory_allocated"], (int, float)):
                st.metric("사용 중인 VRAM", f"{gpu_info['memory_allocated']} GB")
            else:
                st.write("**사용 중인 VRAM**: 확인 불가")

            if isinstance(gpu_info["memory_total"], (int, float)):
                st.metric("전체 VRAM", f"{round(gpu_info['memory_total'], 1)} GB")
            else:
                st.write("**전체 VRAM**: 확인 불가")

        # Whisper 모델의 GPU 사용 설정
        st.info(f"Whisper 모델 확인: GPU 사용이 가능합니다! {gpu_info['device_name']}에서 사용률이 낮은 경우 GPU 부하가 낮거나 CPU로 일부 작업이 처리될 수 있습니다.")

        # 환경 변수 확인 - expander 사용하지 않고 직접 표시
        st.subheader("GPU 최적화 설정 확인")
        cuda_env_vars = {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "설정되지 않음"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "설정되지 않음"),
            "TF_FORCE_GPU_ALLOW_GROWTH": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "설정되지 않음")
        }

        for var_name, var_value in cuda_env_vars.items():
            st.write(f"**{var_name}**: {var_value}")

        if all(value == "설정되지 않음" for value in cuda_env_vars.values()):
            st.warning("GPU 관련 환경 변수가 설정되지 않았습니다. 필요한 경우 최적화를 위해 환경 변수를 설정하세요.")

        # Torch 버전 정보
        st.write(f"**PyTorch 버전**: {torch.__version__}")
        st.write(f"**CUDA 버전**: {torch.version.cuda or '사용 불가'}")
    else:
        st.warning("GPU가 감지되지 않았습니다. CPU 모드로 실행됩니다.")
        st.write("Whisper 모델은 CPU에서도 작동하지만, 처리 속도가 느립니다.")

        # 가능한 원인 및 해결책 - expander 없이 직접 표시
        st.subheader("가능한 원인 및 해결책")
        st.write("""
        - **CUDA가 설치되지 않음**: PyTorch CUDA 버전을 설치하세요: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
        - **드라이버 문제**: 최신 NVIDIA 드라이버가 설치되어 있는지 확인하세요.
        - **CUDA 버전 불일치**: PyTorch와 호환되는 CUDA 버전을 설치하세요.
        - **환경 변수 문제**: 'CUDA_VISIBLE_DEVICES' 환경 변수가 올바르게 설정되어 있는지 확인하세요.
        """)
