param([int]$Port = 8765, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$python = $env:LABELPILOT_PYTHON
if (-not $python) { $python = 'D:\Miniconda3\envs\key_action_py311\python.exe' }
if (-not (Test-Path -LiteralPath $python)) { $python = (Get-Command python -ErrorAction Stop).Source }
$url = "http://127.0.0.1:$Port"
try {
    $existing = Invoke-RestMethod "$url/api/health" -TimeoutSec 2
    if ($existing.app -eq 'LabelPilot') {
        if (-not $NoBrowser) { Start-Process $url }
        Write-Host "LabelPilot is already running: $url"
        exit 0
    }
    throw 'This port is used by another application.'
} catch {
    if ($_.Exception.Message -like '*This port*') { throw }
}
Write-Host "LabelPilot | $url | Ctrl+C to stop"
Write-Host "Python: $python"
if (-not $NoBrowser) {
    $browserJob = Start-Job -ArgumentList $url -ScriptBlock {
        param($address)
        for ($attempt = 0; $attempt -lt 90; $attempt++) {
            try {
                $health = Invoke-RestMethod "$address/api/health" -TimeoutSec 2
                if ($health.app -eq 'LabelPilot') { Start-Process $address; return }
            } catch { }
            Start-Sleep -Seconds 1
        }
    }
}
try {
    & $python -X utf8 (Join-Path $PSScriptRoot 'run.py') --port $Port
    $result = $LASTEXITCODE
} finally {
    if ($browserJob) { Stop-Job $browserJob -ErrorAction SilentlyContinue; Remove-Job $browserJob -Force -ErrorAction SilentlyContinue }
}
exit $result
