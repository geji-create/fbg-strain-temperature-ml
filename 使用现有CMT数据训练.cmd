@echo off
cd /d "%~dp0"
call "%~dp0run_python.cmd" run_pipeline.py --config config.default.json --input-csv "%~dp0data\source_synthetic_exports\fbg_uniform_cmt_spectra.csv" --output-dir outputs\existing_cmt_baseline
pause

