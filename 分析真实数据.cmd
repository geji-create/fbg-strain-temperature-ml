@echo off
chcp 65001 >nul
cd /d "%~dp0"
call "%~dp0run_python.cmd" analyze_real_data.py --input-dir "data\real_raw" --output-dir "outputs\real_data"
pause
