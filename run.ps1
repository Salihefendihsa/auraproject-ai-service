# PowerShell Run Script - AuraProject AI Service v1.1.0
# Run: .\run.ps1

Write-Host "======================================" -ForegroundColor Cyan
Write-Host " AuraProject AI Service v1.1.0" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Check .venv exists
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "ERROR: .venv not found! Run setup.ps1 first." -ForegroundColor Red
    exit 1
}

# Check API key
if (-not $env:OPENAI_API_KEY) {
    Write-Host ""
    Write-Host "WARNING: OPENAI_API_KEY not set!" -ForegroundColor Yellow
    Write-Host "LLM outfit generation will fail." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host ""
Write-Host "Starting server..." -ForegroundColor Yellow
Write-Host "API: http://localhost:8000" -ForegroundColor Green
Write-Host "Docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host ""

# Start server using .venv
& .\.venv\Scripts\python.exe -m uvicorn ai_service.app.main:app --reload --port 8000
