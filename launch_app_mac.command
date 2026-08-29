#!/bin/bash

APP_DIR="$(cd "$(dirname "$0")" && pwd)" || exit 1
cd "$APP_DIR" || exit 1

APP_SUPPORT_DIR="${HOME}/Library/Application Support/Electrochemistry Analysis Scripts"
VENV_DIR="${APP_SUPPORT_DIR}/venv"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    if command -v python3 >/dev/null 2>&1; then
        SYSTEM_PYTHON="python3"
    elif command -v python >/dev/null 2>&1; then
        SYSTEM_PYTHON="python"
    else
        echo "Python was not found."
        echo "Install Python 3 from https://www.python.org/downloads/ and try again."
        read -r -p "Press Return to close..."
        exit 1
    fi

    echo "Creating a Python virtual environment in:"
    echo "$VENV_DIR"
    if ! mkdir -p "$APP_SUPPORT_DIR"; then
        echo "Failed to create the local application-support folder."
        read -r -p "Press Return to close..."
        exit 1
    fi
    if ! "$SYSTEM_PYTHON" -m venv "$VENV_DIR"; then
        echo "Failed to create the virtual environment."
        read -r -p "Press Return to close..."
        exit 1
    fi
fi

PYTHON="$VENV_DIR/bin/python"

echo "Installing required packages..."
if ! "$PYTHON" -m pip install --disable-pip-version-check -r requirements.txt; then
    echo "Failed to install the required packages."
    read -r -p "Press Return to close..."
    exit 1
fi

echo "Starting the Electrochemistry Analysis app..."
echo "Your browser should open at http://localhost:8501"
"$PYTHON" -m streamlit run app.py --server.headless=false --browser.gatherUsageStats=false
status=$?

if [ "$status" -ne 0 ]; then
    echo
    echo "The app stopped with an error."
    read -r -p "Press Return to close..."
fi

exit "$status"
