@echo off
REM Launch the FPGA fault-classification viewer (Replay + Live FPGA tabs).
REM Replay needs only Python 3 + Tkinter; Live additionally needs ssh/scp + board.
REM Usage: run_fpga_viewer.bat [both^|replay^|live]
setlocal
cd /d "%~dp0"
set MODE=%1
if "%MODE%"=="" set MODE=both
python -m host.fpga_result_viewer --mode %MODE%
endlocal
