@echo off

rem Run pylint on specified Python files/directories
pylint utils.py validation.py visualizations.py main.py

rem Check the exit code of pylint
if %errorlevel% neq 0 (
    rem Pylint failed
    echo build-failed
    exit /b %errorlevel%
)

rem Run pytest to execute tests
python -m pytest

rem Check the exit code of pytest
if %errorlevel% neq 0 (
    rem Pytest failed
    echo build-failed
    exit /b %errorlevel%
)

rem Both pylint and pytest succeeded
echo build-successful
exit /b 0
