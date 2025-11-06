# Setup and Installation Script
# Run this once to set up the development environment

Write-Host "Setting up Pharma List Management Application" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
Write-Host ""

# Backend Setup
Write-Host "Setting up Backend..." -ForegroundColor Cyan
Set-Location "$PSScriptRoot\backend"

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment and install dependencies
Write-Host "Installing backend dependencies..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
deactivate

Write-Host "Backend setup complete!" -ForegroundColor Green
Write-Host ""

# Frontend Setup
Write-Host "Setting up Frontend..." -ForegroundColor Cyan
Set-Location "$PSScriptRoot\pharma-frontend"

Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
npm install

Write-Host "Frontend setup complete!" -ForegroundColor Green
Write-Host ""

# Return to root
Set-Location $PSScriptRoot

Write-Host "==============================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application, run: .\start-dev.ps1" -ForegroundColor Yellow
Write-Host ""
