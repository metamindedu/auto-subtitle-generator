@echo off
chcp 65001 > nul
title 자동 자막 생성기

cd /d "%~dp0"

echo ===================================================
echo         자동 자막 생성기 실행 중...
echo ===================================================
echo.

REM 가상환경의 Python을 직접 사용하여 앱 실행
venv\Scripts\python.exe -m streamlit run app.py

pause
