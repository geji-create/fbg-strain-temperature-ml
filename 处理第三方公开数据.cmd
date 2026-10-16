@echo off
chcp 65001 >nul
cd /d "%~dp0"
call "%~dp0run_python.cmd" scripts\process_external_dual_peak_dataset.py
echo.
echo 第三方公开 FBG 数据处理完成，结果位于 outputs\external_mendeley_dual_peak 和 figures\publication。
pause
