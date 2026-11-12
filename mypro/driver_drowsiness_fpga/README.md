# Driver Drowsiness FPGA

FPGA-Accelerated Lightweight CNN for Driver Drowsiness and Distraction Monitoring System Using Seven Keypoints (Nose, Eyes, Ears and Shoulders) on the EDGE Z7-10 Zynq FPGA Development Board.

## Project Identity

| Item | Value |
|------|-------|
| Programme | M.Tech — VLSI and Embedded Systems |
| Target board | EDGE Z7-10 Zynq FPGA Development Board |
| SoC | Xilinx Zynq-7000 (exact part TBD from physical board) |
| Model | `DriverPoseNet7KP` |
| Input / Output | `[B, 3, 256, 192]` → `[B, 7, 64, 48]` |
| Keypoint order | 0:Nose, 1:Left Eye, 2:Right Eye, 3:Left Ear, 4:Right Ear, 5:Left Shoulder, 6:Right Shoulder |

## Repository Structure

```
driver_drowsiness_fpga/
├── configs/           # YAML configuration files
├── datasets/          # Dataset symlinks / manifests
├── src/               # Python source (dataset, model, training, ...)
├── scripts/           # Build and run scripts
├── hls/               # Vitis HLS accelerator modules
├── rtl/               # Hand-written / generated RTL
├── vivado/            # Vivado project / block design
├── app/               # Embedded real-time demo
├── results/           # Training, accuracy, quantization, HLS, Vivado, hardware
├── docs/              # Documentation and environment report
├── tests/             # Unit tests
├── requirements.txt
└── .gitignore
```

## Getting Started

See `docs/environment_report.md` for the Stage 0 laptop/FPGA audit.

## Citation / Reference

Baseline paper: Minjoon Kim and Jaehyuk So, "Real-Time Driver Monitoring: Implementing FPGA-Accelerated CNNs for Pose Detection," IEEE TVLSI, 2025. DOI: 10.1109/TVLSI.2025.3554880.
"# driver_drowsiness_fpga" 
