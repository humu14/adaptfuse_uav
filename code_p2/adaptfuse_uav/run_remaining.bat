@echo off
cd /d "%~dp0"
call C:\Users\USERAS\anaconda3\Scripts\activate.bat C:\Users\USERAS\anaconda3
powershell.exe -ExecutionPolicy Bypass -File .\run_remaining.ps1
pause
