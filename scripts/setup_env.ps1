# PowerShell helper to create venv and install requirements
param(
    [string]$venvName = "venv"
)

python -m venv $venvName
$activate = Join-Path $PSScriptRoot $venvName
# Activate differs across PowerShell versions; inform user
Write-Host "Created virtual environment: $venvName"
Write-Host "To activate (PowerShell): .\\$venvName\\Scripts\\Activate.ps1"
Write-Host "Then run: pip install -r \"c:\Users\PC\OneDrive\Desktop\AI Deepfake Detector\requirements.txt\""

# Optionally install PyTorch with CUDA 11.8 (uncomment to automate)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
