@echo off
cd /d "%~dp0"
call "%~dp0run_python.cmd" run_cnn.py --input-csv "%~dp0data\source_synthetic_exports\fbg_uniform_cmt_spectra.csv" --output-dir outputs\cnn_baseline
pause

