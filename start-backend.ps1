# PowerShell script to start the backend server
Set-Location $PSScriptRoot
Set-Location backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
