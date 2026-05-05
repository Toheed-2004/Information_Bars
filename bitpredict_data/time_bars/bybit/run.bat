@echo off
setlocal

REM ------------------------------
REM Conda configuration
REM ------------------------------
set "CONDAPATH=%CONDAPATH%"
set "ENVNAME=bitpredict"

if "%ENVNAME%"=="base" (
    set "ENVPATH=%CONDAPATH%"
) else (
    set "ENVPATH=%CONDAPATH%\envs\%ENVNAME%"
)

REM ------------------------------
REM Activate Conda environment ONCE
REM ------------------------------
call "%CONDAPATH%\Scripts\activate.bat" "%ENVPATH%"

REM Ensure correct working directory
cd /d "%~dp0"

REM ------------------------------
REM Run processes in parallel
REM ------------------------------
start "INIT"     cmd /c python main.py init
start "UPDATE"   cmd /c python main.py update

start "RESAMPLE" cmd /c python main.py resample

