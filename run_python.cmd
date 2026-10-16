@echo off
setlocal
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Python environment not found. Create .venv in this folder or its parent folder.
  exit /b 1
)
"%PYTHON%" %*
exit /b %errorlevel%
