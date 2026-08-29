@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    where py >nul 2>&1
    if not errorlevel 1 (
        set "SYSTEM_PYTHON=py"
    ) else (
        where python >nul 2>&1
        if errorlevel 1 (
            echo Python was not found.
            echo Install Python 3 from https://www.python.org/downloads/ and try again.
            pause
            exit /b 1
        )
        set "SYSTEM_PYTHON=python"
    )

    echo Creating a Python virtual environment...
    !SYSTEM_PYTHON! -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

set "PYTHON=.venv\Scripts\python.exe"

echo Installing required packages...
"%PYTHON%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo Failed to install the required packages.
    pause
    exit /b 1
)

echo Starting the Electrochemistry Analysis app...
echo Your browser should open at http://localhost:8501
"%PYTHON%" -m streamlit run app.py --server.headless=false --browser.gatherUsageStats=false

if errorlevel 1 (
    echo.
    echo The app stopped with an error.
    pause
)

endlocal
