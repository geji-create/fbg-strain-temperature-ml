@echo off
cd /d "%~dp0"
call "%~dp0run_python.cmd" run_pipeline.py --config config.default.json --output-dir outputs\quick_check --quick
pause

