# PowerShell Setup Script - AuraProject AI Service v1.1.0
# Run: .\setup.ps1

Write-Host "======================================" -ForegroundColor Cyan
Write-Host " AuraProject AI Service v1.1.0 Setup" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Check Python
$pythonCheck = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] $pythonCheck" -ForegroundColor Green

# Create .venv (not venv)
if (-not (Test-Path ".venv")) {
    Write-Host "[..] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
}
Write-Host "[OK] Virtual environment ready (.venv)" -ForegroundColor Green

# Install dependencies
Write-Host "[..] Installing dependencies..." -ForegroundColor Yellow
& .\.venv\Scripts\pip.exe install -r requirements.txt --quiet
Write-Host "[OK] Dependencies installed" -ForegroundColor Green

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host " Setup Complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "NOTE: First run will download SegFormer model (~300MB)" -ForegroundColor Yellow
Write-Host ""
Write-Host "Set your OpenAI API key:" -ForegroundColor Cyan
Write-Host '  $env:OPENAI_API_KEY = "sk-..."' -ForegroundColor White
Write-Host ""
Write-Host "Then run: .\run.ps1" -ForegroundColor Cyan
