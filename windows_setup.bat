@echo off
echo Currently in %~dp0
echo.
echo This script will require administrator permissions to set an environment variable so that the main script can run.
echo.
pause
SETX BRAINART %~dp0
echo Environment Variable "BRAINART" set to %~dp0
pause

:: Just in case make sure that we stay in the right directory if we are in administrator mode
cd %~dp0

:: Check if .esp directory exists
if not exist ".esp" (
    echo Creating .esp directory...
    python -m venv .esp
)

:: Activate the virtual environment
call .esp\Scripts\activate.bat

:: Install required packages
echo Installing required packages...
python -m pip install -r requirements.txt
echo Packages installed.
pause

:: Deactivate the virtual environment
echo Deactivating virtual environment. Click enter to exit.
pause
deactivate