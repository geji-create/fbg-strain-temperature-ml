@echo off
chcp 65001 >nul
cd /d "%~dp0"
call "%~dp0run_python.cmd" run_hybrid.py --input-dir "data\real_raw" --config "hybrid_config.gradient8.json" --output-dir "outputs\hybrid_digital_twin" --skip-cnn
pause
