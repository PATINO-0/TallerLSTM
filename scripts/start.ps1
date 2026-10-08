param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear el entorno. Instala Python 3.12 o 3.13.' }
}
& $taskPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'No se pudieron instalar las dependencias.' }
Write-Host "Aplicacion: http://127.0.0.1:$Port"
Write-Host "API: http://127.0.0.1:$Port/docs"
& $taskPython -m uvicorn app:app --host 127.0.0.1 --port $Port
