# Clone the official Wavelet-CLIP repository into ./wavelet-clip
param(
    [string]$targetDir = "wavelet-clip"
)

if (Test-Path $targetDir) {
    Write-Host "Directory $targetDir already exists. Skipping clone."
    exit 0
}

Write-Host "Cloning Wavelet-CLIP into $targetDir..."
git clone https://github.com/lalithbharadwajbaru/wavelet-clip.git $targetDir
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to clone wavelet-clip. Please clone manually."
    exit 1
}
Write-Host "Cloned to $targetDir"