# Stage 0 — Laptop & FPGA Environment Report

**Project:** FPGA-Accelerated Lightweight CNN for Driver Drowsiness and Distraction Monitoring (DriverPoseNet7KP)  
**Target board:** EDGE Z7-10 Zynq FPGA Development Board  
**Report date:** 2026-11-12  
**Status:** PARTIAL — software audit complete; FPGA board physical verification pending

---

## 1. Laptop Environment

| Item | Value | Source |
|------|-------|--------|
| Operating system | Windows 11 Pro (build 10.0.26300) | `Get-ComputerInfo` |
| Processor | Intel Core i7-8550U, 4 cores / 8 threads, ~1.99 GHz | `Get-CimInstance Win32_Processor` |
| RAM (GB) | 16 GB (2 × 8 GB, 2400 MHz) | `Get-CimInstance Win32_PhysicalMemory` |
| GPU | Intel UHD Graphics 620 (1 GB) + AMD Radeon R7 M460 (2 GB) | `Get-CimInstance Win32_VideoController` |
| VRAM (GB) | Intel UHD: 1 GB shared; Radeon R7 M460: 2 GB | `Get-CimInstance Win32_VideoController` |
| Python version | 3.13.9 | `python --version` |
| pip | 25.1 | `python -m pip --version` |
| Git | NOT INSTALLED | `git --version` |
| Install action | Download Git for Windows from https://git-scm.com/download/win | Required for version control |
| PyTorch | 2.10.0+cpu (CPU only) | `import torch` |
| Torchvision | 0.25.0+cpu | `import torchvision` |
| OpenCV | 5.0.0 (installed) | `import cv2` |
| NumPy | 2.1.3 | `import numpy` |
| ONNX | 1.23.2 (installed) | `import onnx` |
| CUDA | Not available | `torch.cuda.is_available()` |
| cuDNN | Not available | N/A |

> Run the read-only PowerShell commands in `docs/STAGE_0_INSPECT_COMMANDS.md` (or re-run the commands from the audit) and fill in the TBD values above.

---

## 2. FPGA / Vitis / Vivado Toolchain

| Item | Value | Source |
|------|-------|--------|
| Vivado installed | INSTALLED | `D:\Xilinx\2025.2\Vivado\bin\vivado.bat` |
| Vivado version | v2025.2 (64-bit), SW Build 6299465 | `vivado.bat -version` |
| Vitis installed | INSTALLED | `D:\Xilinx\2025.2\Vitis\bin\vitis.bat` |
| Vitis version | v2025.2 (64-bit), SW Build 6295257 | `vitis.bat` |
| Vitis HLS installed | NOT FOUND as separate executable | Integrated into Vitis v2025.2 |
| Xilinx install path | `D:\Xilinx\2025.2\` | `Test-Path D:\Xilinx` |

---

## 3. EDGE Z7-10 Board (PENDING PHYSICAL VERIFICATION)

These items MUST be inspected on the actual board/laptop. Do not assume values.

| Item | Value | Source |
|------|-------|--------|
| Exact Zynq device | XC7Z010 (user-reported) | board silkscreen / Vivado Hardware Manager |
| FPGA part number | xc7z010clg00n-1 | board silkscreen — VERIFIED by user |
| Board revision | TBD | board silkscreen |
| FPGA resources (LUT/FF/DSP/BRAM) | TBD | device datasheet / synthesis report |
| DDR memory | TBD | board spec |
| Clock sources (MHz) | TBD | board spec |
| JTAG status | TBD | Vivado Hardware Manager / `Get-PnpDevice` |
| UART status | TBD | Device Manager / `Get-PnpDevice` |
| Camera interface | TBD | board spec / physical inspection |
| Vivado/Vitis compatibility | TBD | board product page |

---

## 4. Project Directory

| Item | Value |
|------|-------|
| Project root | `G:\mypro\driver_drowsiness_fpga` |
| Dataset images | `G:\mypro\driver_drowsiness_fpga\datasets\coco` (COCO images present) |
| COCO annotations JSON | FOUND — `G:\mypro\driver_drowsiness_fpga\datasets\coco\annotations\` |
| Annotations files | person_keypoints_train2017.json, person_keypoints_val2017.json, instances_train2017.json, instances_val2017.json, captions_train2017.json, captions_val2017.json |
| MASTER_PROJECT_PROMPT.md | `G:\mypro\driver_drowsiness_fpga\MASTER_PROJECT_PROMPT.md` (present) |

---

## 5. Discrepancies vs. Prompt Assumptions

- COCO dataset location `G:\mypro\driver_drowsiness_fpga\datasets\coco` exists with images AND annotations JSON (person_keypoints_train2017.json, person_keypoints_val2017.json).
- OpenCV (5.0.0) and ONNX (1.23.2) have been installed; PyYAML, Matplotlib, SciPy, Tqdm were already present.
- **Git is NOT installed** — required for version control per project rules. Download Git for Windows from https://git-scm.com/download/win.
- Vivado v2025.2 and Vitis v2025.2 INSTALLED on D: drive (`D:\Xilinx\2025.2\`). Vitis HLS not found as separate executable (likely integrated into Vitis).
- Zynq part number confirmed: xc7z010clg00n-1 (XC7Z010). Matches the XC7Z010 assumption in the prompt.
- CUDA not available; PyTorch runs CPU-only. Training will be slow on this laptop.

---

## 6. Next Action

1. Install Git for Windows from https://git-scm.com/download/win (required for version control).
2. Initialize Git repository in `G:\mypro\driver_drowsiness_fpga` and make first commit.
3. Proceed to Stage 1 (Research and architecture) — all software prerequisites are now met.
4. FPGA board physical verification (board revision, JTAG, UART, camera interface) still pending but does not block Stage 1.
