@echo off
REM Ingest the 4-video demo corpus (2 MIT + 2 Stanford) in one go.
REM
REM Usage:
REM   scripts\ingest_demo.bat
REM   scripts\ingest_demo.bat --force
REM
REM This is a thin wrapper around scripts/ingest_demo.py — use the Python
REM version (`python -m scripts.ingest_demo`) if you want the --videos flag
REM to ingest a subset.

setlocal
cd /d "%~dp0\.."
python -m scripts.ingest_demo %*
endlocal
