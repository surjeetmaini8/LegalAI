Write-Host "`n🔍 Checking MongoDB Installation..." -ForegroundColor Cyan

$mongoService = Get-Service -Name "MongoDB" -ErrorAction SilentlyContinue

if ($mongoService) {
    Write-Host "MongoDB service found!" -ForegroundColor Green
    Write-Host "   Status: $($mongoService.Status)" -ForegroundColor Yellow
    
    if ($mongoService.Status -ne "Running") {
        Write-Host "`n Starting MongoDB service..." -ForegroundColor Cyan
        Start-Service -Name "MongoDB"
        Start-Sleep -Seconds 3
        Write-Host "MongoDB service started!" -ForegroundColor Green
    }
} else {
    Write-Host "MongoDB service not found" -ForegroundColor Red
    Write-Host "   Please install MongoDB from: https://www.mongodb.com/try/download/community" -ForegroundColor Yellow
}

# Test MongoDB connection
Write-Host "`n Testing MongoDB connection..." -ForegroundColor Cyan
Test-NetConnection -ComputerName localhost -Port 27017 -InformationLevel Detailed
