@echo off
setlocal
chcp 65001 >nul
set "TASK_PYTHON=%~dp0..\..\.venv\Scripts\python.exe"
if not exist "%TASK_PYTHON%" set "TASK_PYTHON=python"
pushd "%~dp0"
"%TASK_PYTHON%" -X utf8 "%~dp0run.py" %*
set "TASK_EXIT=%ERRORLEVEL%"
popd
exit /b %TASK_EXIT%
