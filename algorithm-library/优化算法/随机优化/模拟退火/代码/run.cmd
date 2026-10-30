@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0"
set "ALGORITHM_PY=%~dp0..\..\.venv\Scripts\python.exe"
if not exist "%ALGORITHM_PY%" set "ALGORITHM_PY=python"
if "%~1"=="" (
    "%ALGORITHM_PY%" -X utf8 run.py demo
    pause
) else (
    "%ALGORITHM_PY%" -X utf8 run.py %*
)
set "ALGORITHM_EXIT=%ERRORLEVEL%"
popd
exit /b %ALGORITHM_EXIT%
