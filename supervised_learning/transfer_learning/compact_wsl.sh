#!/bin/bash

# compact_wsl.sh
# This script is designed to compact the WSL disk on Windows.
## NOTICE, this file assumes that you have a PowerShell script that is
## located at /mnt/c/Scripts/compact_wsl.ps1
##
##
####
##
## PowerShell script:
##  $vhdPath = Join-Path $wslPath "ext4.vhdx"
##  
##  # Create diskpart script
##  $diskpartScript = @"
##  select vdisk file="$vhdPath"
##  attach vdisk readonly
##  compact vdisk
##  detach vdisk
##  exit
##  "@
## 
##  # Save diskpart script to temporary file
##  $tempFile = [System.IO.Path]::GetTempFileName()
##  $diskpartScript | Out-File -FilePath $tempFile -Encoding ASCII
##
##  #Run diskpart with the script
##  Write-Host "Running diskpart to compact the virtual disk..."
##  diskpart /s $tempFile
##
##  #Clean up
##  Remove-Item $tempFile
##
##Write-Host "Compaction completed. You can now restart WSL."
##
##
####


#Path to the PowerShell script on Windows
POWERSHELL_SCRIPT="/mnt/c/Scripts/compact_wsl.ps1"  # Adjust this path as needed

echo "Preparing to compact WSL disk..."
echo "This operation requires Windows administrator privileges."

# Convert the Linux path to a Windows path for powershell.exe
WINDOWS_PATH=$(wslpath -w "$POWERSHELL_SCRIPT")

# Execute the PowerShell script using powershell.exe
powershell.exe -Command "Start-Process powershell -ArgumentList '-ExecutionPolicy Bypass -File $WINDOWS_PATH' -Verb RunAs"

echo "Compaction process initiated. The WSL instance will shut down."
echo "Please wait for the PowerShell window to complete the operation."