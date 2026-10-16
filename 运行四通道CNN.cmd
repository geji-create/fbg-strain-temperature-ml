@echo off
chcp 65001 >nul
cd /d "%~dp0"
call "%~dp0run_python.cmd" run_multichannel_cnn.py --spectra "outputs\hybrid_digital_twin\hybrid_spectra.csv.gz" --real-anchor "outputs\hybrid_digital_twin\real_anchor_spectra.csv.gz" --config "hybrid_config.gradient8.json" --output-dir "outputs\hybrid_digital_twin\multichannel_cnn"
pause
