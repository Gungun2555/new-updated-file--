# Start both frontend and backend concurrently
# Usage: .\start-dev.ps1

Write-Host "Starting Pharma List Management - Development Mode" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
Write-Host ""

# Start backend in a new window
Write-Host "Starting FastAPI Backend..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\backend'; Write-Host 'Backend Server Running on http://localhost:8000' -ForegroundColor Yellow; uvicorn app.main:app --reload --port 8000"

# Wait a moment for backend to start
Start-Sleep -Seconds 2

# Start frontend in a new window
Write-Host "Starting React Frontend..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\pharma-frontend'; Write-Host 'Frontend Running on http://localhost:5173' -ForegroundColor Yellow; npm run dev"

Write-Host ""
Write-Host "Both services are starting..." -ForegroundColor Green
Write-Host "Backend: http://localhost:8000" -ForegroundColor Yellow
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Yellow
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
