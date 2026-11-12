# Stage 1 — Research and Architecture

**Project:** FPGA-Accelerated Lightweight CNN Driver Monitoring (DriverPoseNet7KP)  
**Target board:** EDGE Z7-10 Zynq (XC7Z010, xc7z010clg00n-1)  
**Status:** COMPLETE

---

## 1. Objective

Stage 1 establishes the research foundation:
1. Review the baseline reference paper and identify the technical gap.
2. Define the proposed DriverPoseNet7KP architecture.
3. Establish measurable objectives and research contributions.
4. Document the architecture in `docs/architecture.md`.

---

## 2. Baseline Reference Paper

| Field | Value |
|-------|-------|
| Title | Real-Time Driver Monitoring: Implementing FPGA-Accelerated CNNs for Pose Detection |
| Authors | Minjoon Kim and Jaehyuk So |
| Publication | IEEE Transactions on VLSI Systems, 2025 |
| DOI | 10.1109/TVLSI.2025.3554880 |

**Important:** The paper PDF has not yet been inspected. Per project rules:
- The paper is a reference, not a completed project.
- Paper results must not be reported as project results.
- Published results, proposed architecture, simulation results, and FPGA measurements must be clearly distinguished.
- Do not invent information absent from the paper.

**Next:** Obtain the paper PDF and inspect it before finalizing the baseline comparison.

---

## 3. Proposed Research Contribution

The proposed system reduces the pose representation from eleven keypoints to **seven keypoints**:

| Index | Keypoint |
|-------|----------|
| 0 | Nose |
| 1 | Left Eye |
| 2 | Right Eye |
| 3 | Left Ear |
| 4 | Right Ear |
| 5 | Left Shoulder |
| 6 | Right Shoulder |

This order is enforced in `src/model/keypoints.py` and must be used consistently across dataset, model, training, decoding, HLS, RTL, and embedded code.

**Research hypothesis:**
> A carefully optimized seven-keypoint CNN, combined with temporal pose analysis and FPGA acceleration, may reduce computational and hardware requirements while retaining sufficient information for driver monitoring.

This is a hypothesis to be proven or rejected through experiments — not an established result.

---

## 4. Proposed Architecture: DriverPoseNet7KP

### 4.1 Model specification

| Item | Value |
|------|-------|
| Model name | `DriverPoseNet7KP` |
| Input | `[B, 3, 256, 192]` (BGR/RGB frame) |
| Output | `[B, 7, 64, 48]` (7 heatmaps, one per keypoint) |
| Heatmap resolution | 64 × 48 |
| Keypoint channels | 7 |
| Gaussian sigma | 2.0 (configurable) |

### 4.2 Architecture diagram

```
Input: [B, 3, 256, 192]
   |
   v
Stem Conv 3x3, s=2: 3->32  -> [B, 32, 128, 96]
Stem BN + ReLU
Stem MaxPool 3x3, s=2 -> [B, 32, 64, 48]
   |
   v
Residual Block 32->64, s=2  -> [B, 64, 32, 24]
   |
   v
Residual Block 64->128, s=2 -> [B, 128, 16, 12]
   |
   v
Residual Block 128->256, s=2 -> [B, 256, 8, 6]
   |
   v
ConvTranspose 256->128, s=2 -> [B, 128, 16, 12]
   |
   v
ConvTranspose 128->64, s=2  -> [B, 64, 32, 24]
   |
   v
ConvTranspose 64->32, s=2   -> [B, 32, 64, 48]
   |
   v
1x1 Conv 32->7              -> [B, 7, 64, 48]
   |
   v
Output: 7 heatmaps
```

### 4.3 Building blocks

- Convolution (3x3, 1x1)
- Batch Normalization
- ReLU activation
- Max pooling (stem only)
- Residual blocks (with 1x1 projection for stride/channel change)
- Transposed convolution (upsampling path)
- Final 1x1 heatmap head (7 output channels)

### 4.4 FPGA-friendly design choices

| Design choice | Reason |
|---------------|--------|
| Stride-2 downsampling | Reduces spatial resolution early, lowering MACs in deeper layers |
| 1x1 projection convolutions | Efficient channel adaptation in residual blocks |
| INT8 quantization support | Configurable in `configs/model.yaml` (8-bit weights/activations, 32-bit accumulator) |
| Transposed convolution | Simple upsampling, implementable in HLS |
| 7-channel output head | Minimal output bandwidth to PS |

---

## 5. Measured Model Complexity (Actual Results)

Measured by running the model in PyTorch (CPU, batch=1):

| Metric | Value | Source |
|--------|-------|--------|
| Total parameters | 1,595,591 (~1.6M) | `sum(p.numel())` |
| Trainable parameters | 1,595,591 | All parameters trainable |
| FP32 model size | 6.10 MB | Parameters + buffers |
| Approx MACs (forward) | 143,425,536 (~143M) | Conv2d MAC counting |
| Approx GFLOPs | 0.287 | MACs × 2 / 1e9 |
| Input shape | `[1, 3, 256, 192]` | Verified |
| Output shape | `[1, 7, 64, 48]` | Verified |

**Counting convention:** GFLOPs = 2 × MACs / 1e9 (each MAC = 1 multiply + 1 add = 2 FLOPs).

**Note:** These are PyTorch simulation results, not FPGA synthesis results. Actual hardware MACs, DSP usage, and latency will differ after HLS implementation and Vivado synthesis.

---

## 6. Technical Gap vs. Baseline

The baseline paper (Kim & So, 2025) uses eleven keypoints on a larger FPGA platform (VCU118-class resources assumed). The proposed work differs in:

1. **Keypoint count:** 7 vs 11 — reduced output head and pose representation.
2. **Target hardware:** EDGE Z7-10 (XC7Z010, ~53K LUTs, ~220 DSPs) — smaller embedded FPGA vs. data-center-class board.
3. **Resource-aware design:** DSP sharing, MAC reuse, tiled convolution, INT8 arithmetic, line buffers — required for the smaller device.

**Pending:** Baseline paper inspection to fill in specific architecture details and published metrics for comparison.

---

## 7. Measurable Objectives (Stage 1)

| Objective | Metric | Target | Status |
|-----------|--------|--------|--------|
| Model input/output shape | `[B, 3, 256, 192]` → `[B, 7, 64, 48]` | Exact match | Verified |
| Seven-keypoint order | 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LSh, 6:RSh | Enforced in code | Verified |
| Parameter count | Measured | < 5M (lightweight) | 1.6M ✓ |
| FP32 model size | Measured | < 20 MB | 6.10 MB ✓ |
| Baseline paper reviewed | Paper inspected | Complete | Pending PDF |

---

## 8. Acceptance Criteria

- [ ] Baseline paper PDF obtained and inspected
- [ ] DriverPoseNet7KP architecture documented in `docs/architecture.md`
- [ ] Model input/output shapes verified (`[B, 3, 256, 192]` → `[B, 7, 64, 48]`)
- [ ] Model complexity measured (parameters, MACs, GFLOPs, model size)
- [ ] Seven-keypoint order enforced and validated
- [ ] Research hypothesis stated (not claimed as result)
- [ ] Measurable objectives established

---

## 9. Next Action

Proceed to Stage 2 — Dataset preparation:
1. Verify COCO annotations JSON at `G:\mypro\driver_drowsiness_fpga\datasets\coco\annotations\`
2. Implement seven-keypoint mapping from COCO 17-keypoint format
3. Generate and verify Gaussian heatmaps
4. Create dataset visualizations
