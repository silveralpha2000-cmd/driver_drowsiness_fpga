# Stage 0 Inspection Commands (PowerShell)

Run these on your Windows laptop. All are read-only — nothing is modified.

## 1. System info
```powershell
Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, CsTotalPhysicalMemory, CsProcessors
```

## 2. Python and key packages
```powershell
python --version
pip list 2>$null | Select-String -Pattern "torch|torchvision|opencv|numpy|onnx|cuda"
```

## 3. GPU / CUDA
```powershell
nvidia-smi 2>$null
```

## 4. Git
```powershell
git --version
```

## 5. Vivado / Vitis / Vitis HLS
```powershell
Get-Command vivado, vitis, vitis_hls -ErrorAction SilentlyContinue | Select-Object Name, Source, Version
Get-ChildItem "C:\Xilinx" -Recurse -Depth 2 -ErrorAction SilentlyContinue | Select-Object FullName | Select-Object -First 20
```

## 6. FPGA board / JTAG / UART devices
```powershell
Get-PnpDevice | Where-Object {$_.FriendlyName -like "*Xilinx*" -or $_.FriendlyName -like "*JTAG*" -or $_.FriendlyName -like "*UART*" -or $_.FriendlyName -like "*Serial*"} | Select-Object FriendlyName, Status, Class
```

## 7. Fill environment_report.md

Paste the outputs above into `docs/environment_report.md`, replacing each "TBD". Then identify the exact Zynq part on the EDGE Z7-10 (board silkscreen or Vivado Hardware Manager).
