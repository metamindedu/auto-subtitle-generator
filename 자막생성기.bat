@echo off
chcp 65001 > nul
title 자동 자막 생성기

cd /d "%~dp0"

echo ===================================================
echo         자동 자막 생성기 실행 중...
echo ===================================================
echo.

REM 가상환경 활성화 및 앱 실행
call venv\Scripts\activate.bat
python -m streamlit run app.py

pause
