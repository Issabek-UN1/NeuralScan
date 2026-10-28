# NeuralScan: Edge-AI Hybrid 3D Metrology Scanner

<div align="center">

[![Language: English](https://img.shields.io/badge/Language-English-blue.svg?style=for-the-badge)](README.md)
[![Language: Russian](https://img.shields.io/badge/Language-Русский-red.svg?style=for-the-badge)](README_RU.md)
[![ASUS UGen AI LEAGUE](https://img.shields.io/badge/Contest-ASUS%20UGen%20AI%20LEAGUE-ff0055.svg?style=for-the-badge&logo=asus)](https://contest.bhuntr.com/en/39jg9vimiynrhlksze/home/)
[![Hardware Rig](https://img.shields.io/badge/Hardware%20Cost-%2430%20USD%20%7C%20Open--Source-00b4d8.svg?style=for-the-badge&logo=arduino)](firmware/)
[![AI Engine](https://img.shields.io/badge/AI%20Core-Depth%20Anything%20V2%20%7C%20CUDA%20FP16-76b900.svg?style=for-the-badge&logo=nvidia)](ai/)
[![Test Suite](https://img.shields.io/badge/Test%20Suite-7%2F7%20Passed%20(100%25)-brightgreen.svg?style=for-the-badge)](test_pipeline.py)
[![CAD Output](https://img.shields.io/badge/CAD%20Export-Watertight%20STL%20%2F%20PLY-orange.svg?style=for-the-badge)](output/models/)

**Democratizing High-Precision 3D Metrology: $30 Open-Source Hardware + Edge-AI on ASUS Platform.**

[20-Page Proposal Deck](docs/asus_proposal_deck_20_pages.md) • [Pitch Deck Script](docs/pitch_deck_script.md) • [Presentation Guide](docs/presentation_guide.md) • [Quickstart](#5-quickstart-guide) • [ASUS Hardware Benchmark](#4-asus-hardware-acceleration-benchmark) • [Русская версия](README_RU.md)

</div>

---

## 1. Executive Summary & Problem Statement

### The Industrial Barrier
Industrial 3D optical metrology scanners (e.g., GOM ATOS, Creaform HandySCAN, Keyence) cost between **$5,000 and $25,000 USD**. Their prohibitive cost is driven by specialized telecentric lenses, dual-camera stereo rigs, calibrated structured-light projectors, and high-resolution optical encoders.

Existing accessible alternatives suffer from fatal trade-offs:
* **Classical Photogrammetry** (Meshroom, RealityCapture) requires hundreds of photos, takes 15–40 minutes of heavy compute, has **no physical metric scale**, and **completely fails on textureless, monochrome, or smooth parts** (such as white plastic cylinders, machined gears, or aluminum flanges).
* **Budget Laser Triangulation Scanners** (e.g., legacy BQ Ciclop) scan point-by-point or line-by-line, taking 30–45 minutes per scan and producing sparse, noisy point clouds.

### The NeuralScan Breakthrough
**NeuralScan** eliminates the need for expensive physical optics by transferring optical complexity into **local Edge-AI running directly on ASUS hardware**:
1. **$30 Open-Hardware Rig:** Operates on an Arduino Uno R3, an off-the-shelf 28BYJ-48 geared stepper motor (with ULN2003 driver), a KY-008 line laser projector ($1.20), and any standard smartphone camera (1080p).
2. **Monocular Foundation Depth (Depth Anything V2):** Instantly predicts continuous dense surface geometry from single viewpoints in native CUDA FP16.
3. **OLS Metric Alignment Layer (Core Novelty):** Resolves the fundamental scale ambiguity ($D_{\text{AI}} \in [0, 1]$) of monocular depth models by anchoring the neural relief to true physical millimeters using an analytical subpixel laser tracker (<1 ms latency).
4. **Guaranteed Watertight CAD Export:** Reconstructs closed 2-manifold surfaces using Screened Poisson Surface Reconstruction and topological manifold surgery (`pymeshfix`), producing 3D-printable binary STL models with zero manual editing.

---

## 2. Hardware Architecture & $30 Bill of Materials (BOM)

```
                 +-------------------------------------------------+
                 |            ASUS PC / Edge AI Host               |
                 | (NVIDIA RTX 4060 / Intel NPU, PyTorch CUDA FP16)|
                 +-----------------------+-------------------------+
                                         |
                       +-----------------+-----------------+
                       | USB UART (115200)                 | USB / Wi-Fi Video (1080p)
                       v                                   v
            +----------------------+            +----------------------+
            |    Arduino Uno R3    |            |  Smartphone / Webcam |
            |    + ULN2003 Board   |            |  (Fixed Focus/Exp)   |
            +----------+-----------+            +----------+-----------+
                       |                                   |
             +---------+---------+                         |
             |                   |                         v
             v                   v             [Target Object @ 300 mm]
    +-----------------+ +-----------------+                ^
    | 28BYJ-48 Motor  | |  KY-008 Laser   |                |
    | (2048 steps/rev)| | Line Projector  |----------------+
    | Turntable Stage | | (650nm Red Line)| (Laser Stripe)
    +-----------------+ +-----------------+
```

### Bill of Materials (BOM)

| Item | Component | Specification | Approximate Cost |
| :---: | :--- | :--- | :---: |
| **1** | **Microcontroller** | Arduino Uno R3 (ATmega328P, USB-UART CH340) | **$4.50** |
| **2** | **Rotary Actuator** | 28BYJ-48 Geared Stepper (64:1 reduction, 2048 steps/rev) | **$2.20** |
| **3** | **Driver Board** | ULN2003 Darlington Transistor Array | **$0.80** |
| **4** | **Optical Emitter** | KY-008 Red Line Projector (650 nm, 5 mW, TTL switchable) | **$1.20** |
| **5** | **Line Diffuser** | 120° Cylindrical Powell Lens | **$0.80** |
| **6** | **Optical Sensor** | Smartphone Camera via Iriun Webcam / 1080p USB Camera | **$0.00** (Existing) |
| **7** | **Rig Chassis** | 3D-Printed PLA Turntable Base & Bearing (~160g PLA filament) | **$3.50** |
| **8** | **Wiring & Power** | Standard USB Cables & Jumper Wires | **$2.50** |
| **TOTAL** | **Full 3D Metrology Rig** | **100% Functional Open-Hardware Scanner** | **$15.50 – $30.00 USD** |

> **Thermal Protection Mechanism:** Stepper coils are actively de-energized (`digitalWrite(8..11, LOW)`) immediately after movement in [`firmware/uln2003_stepper_controller/`](firmware/uln2003_stepper_controller/). This eliminates resting heat dissipation, keeping the motor and driver at room temperature (<30°C) during continuous scanning.

---

## 3. Mathematical Foundations & Algorithms

```mermaid
flowchart TD
    A["Dual-Frame Capture<br>I_ambient & I_laser"] --> B["Differential Image<br>I_diff = I_laser - I_ambient"]
    B --> C["Subpixel Peak Extraction<br>Weighted COM (gamma=3)<br>Latency < 1 ms"]
    C --> D["Optical Triangulation<br>Z_metric along laser stripe"]
    
    A --> E["u2net Segmentation<br>Foreground Mask (Mask_object)"]
    A --> F["Depth Anything V2<br>Relative Depth D_AI"]
    
    D --> G["OLS Metric Alignment Layer<br>Regression: Z = s * D_AI + t"]
    F --> G
    E --> G
    
    G --> H["Dense 3D Metric Cloud<br>(up to 150,000 points/view)"]
    H --> I["Turntable Kinematics<br>R_y(-theta) + SOR Filter"]
    I --> J["Screened Poisson Reconstruction<br>+ Manifold Repair (pymeshfix)"]
    J --> K["Watertight CAD STL<br>Ready for 3D Printing"]
```

### 3.1. Analytical Subpixel Laser Extraction (< 1 ms)
Instead of slow, pixel-quantized object detectors (such as YOLO), we compute the subpixel horizontal centroid $x_c(y)$ along each scanline $y$ in $< 0.8 \text{ ms}$ using a contrast-weighted center-of-mass formulation ($\gamma = 3$):

$$x_c(y) = \frac{\sum_{x \in W} x \cdot (I_{\text{diff}}(x, y))^\gamma}{\sum_{x \in W} (I_{\text{diff}}(x, y))^\gamma}$$

where:
* $I_{\text{diff}} = \max(0, I_{\text{laser\_on}}^{\text{Red}} - I_{\text{ambient}}^{\text{Red}})$ cancels background room illumination.
* $W = [x_{\text{peak}} - R, x_{\text{peak}} + R]$ is a localized window around the peak ($R = 12 \text{ px}$).
* $\gamma = 3$ mathematically suppresses side-lobe diffusion, delivering **0.05 mm spatial accuracy**.

Physical depth along the laser stripe is calculated via optical triangulation:

$$Z_{\text{metric}}(x_c, y) = \frac{b \cdot f}{(x_c - x_0) + f \cdot \tan(\alpha)}$$

* Baseline $b = 160\text{ mm}$, Focal length $f = 1620\text{ px}$, Convergence angle $\alpha = 28.5^\circ$, Principal point $x_0 = 960\text{ px}$.

### 3.2. OLS Metric Alignment Layer (Resolving Scale Ambiguity)
Monocular depth models output unitless relative depth $D_{\text{AI}} \in [0, 1]$. To map the entire surface into physical millimeters, we pair points along the laser stripe $\{(D_{\text{AI}}(x_c, y), Z_{\text{metric}}(x_c, y))\}$ and solve an Ordinary Least Squares (OLS) linear regression:

$$s = \frac{\text{Cov}(D_{\text{AI}}, Z_{\text{metric}})}{\text{Var}(D_{\text{AI}})}, \quad t = \bar{Z}_{\text{metric}} - s \cdot \bar{D}_{\text{AI}}$$

Dense metric depth is then computed across all segmented foreground pixels:

$$Z_{\text{dense}}(x, y) = s \cdot D_{\text{AI}}(x, y) + t \quad \forall (x, y) \in \text{Mask}_{\text{object}}$$

### 3.3. 3D Kinematics & Watertight Reconstruction
1. **Multi-View Integration:** Point clouds from each angular step $\theta_k$ ($k \in [0, K-1]$) are transformed into canonical coordinates centered at the turntable axis:
   $$\mathbf{P}_{\text{world}} = \mathbf{R}_y(-\theta_k) \cdot (\mathbf{P}_k - \mathbf{C}_{\text{turntable}})$$
2. **Filtering:** Voxel grid filtering (0.8 mm) + Statistical Outlier Removal (SOR, 20 neighbors, $2.0\sigma$).
3. **Screened Poisson Surface Reconstruction:** Generates a continuous, smooth polygonal mesh (Octree depth 8–9).
4. **Watertight Manifold Repair:** Automatic normal winding inversion guard (`if mesh.volume < 0: mesh.invert()`) and topological surgery (`pymeshfix`) guarantee a **100% closed, 2-manifold binary STL**.

---

## 4. ASUS Hardware Acceleration Benchmark

An empirical benchmark was conducted on the workstation host comparing GPU acceleration against standard sequential CPU compute:

```
======================================================================
  ASUS UGen AI LEAGUE: HARDWARE ACCELERATION JUSTIFICATION BENCHMARK
======================================================================
[1/2] NVIDIA RTX 4060 Laptop GPU (CUDA FP16 Tensor Cores):
  -> Latency per frame:   72.17 ms
  -> Throughput:          13.9 FPS (Quasi-Real-Time Metrology)

[2/2] Host CPU Sequential Inference (x86 Float32):
  -> Latency per frame:   1,360.11 ms (1.36 seconds per view)
  -> Throughput:          0.74 FPS (Severe System Bottleneck)
----------------------------------------------------------------------
  ACCELERATION SUMMARY:
  - GPU vs CPU Speedup Ratio:        18.8x FASTER
  - Total 48-step scan on CPU:       65.3 seconds
  - Total 48-step scan on Edge GPU:  3.5 seconds
  - Verdict: Edge GPU acceleration is required for interactive scanning.
======================================================================
```

### Key Advantages of Local Edge AI:
* **Absolute CAD Confidentiality:** 100% on-device processing. No 3D meshes or proprietary intellectual property leave the local machine.
* **Zero Cloud Latency:** Eliminates 30-second network transfer delays for multi-frame high-resolution image sets.
* **Energy-Efficient Deployment:** Architecture is ready for migration to the **ASUS UGen300 AI Accelerator** (Hailo-10H NPU) with under 10W power consumption.

---

## 5. Quickstart Guide

### 5.1. Installation

```powershell
# 1. Clone repository
git clone https://github.com/Issabek-UN1/NeuralScan.git
cd NeuralScan

# 2. Create and activate virtual environment (Python 3.11 recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install pinned dependencies
pip install -r requirements.txt
```

### 5.2. Flash Arduino Firmware
1. Open [`firmware/uln2003_stepper_controller/uln2003_stepper_controller.ino`](firmware/uln2003_stepper_controller/uln2003_stepper_controller.ino) in Arduino IDE.
2. Select Board: **Arduino Uno** (or Mega 2560).
3. Select Port and click **Upload**.
4. **Important:** Close the Arduino IDE Serial Monitor before running Python scripts to release the COM port.

### 5.3. Live Camera Targeting Preview
Before scanning, verify optical framing and laser positioning using the interactive targeting crosshairs:

```powershell
# Open live preview for camera 1 (smartphone / webcam)
python main.py --preview --cam 1
```
*Ensure the object is centered on the green crosshair at approximately 300 mm distance. Press `Q` to exit.*

### 5.4. Run 3D Scan

```powershell
# Fast 8-step scan (45-degree increments) with frame logging:
python main.py --cam 1 --steps 8 --save-frames

# Full high-resolution 48-step scan (7.5-degree increments):
python main.py --cam 1 --steps 48

# Return turntable to origin (0 degrees):
python main.py --home

# Run the official ASUS jury benchmark (CPU vs GPU speed comparison):
python main.py --benchmark
```

### 5.5. Virtual Simulation Mode (Mock Mode)
To test the pipeline without physical hardware connected:
```powershell
python main.py --mock --steps 8
```

---

## 6. Automated Verification Suite (Unit Tests)

All modules are validated with comprehensive unit tests:

```powershell
python test_pipeline.py
```

```
[TEST 1/7] Hardware Serial Controller (Mock Simulation)...    -> PASSED
[TEST 2/7] Camera Stream & Dual-Frame Acquisition...          -> PASSED
[TEST 3/7] Analytical Subpixel Laser Extraction (< 1 ms)...   -> PASSED
[TEST 4/7] Foreground Object Segmentation (u2net)...          -> PASSED
[TEST 5/7] Monocular Relative Depth (Depth Anything V2)...    -> PASSED
[TEST 6/7] Metric Alignment Layer (Scale Resolution)...       -> PASSED
[TEST 7/7] Multi-View Integration & Watertight STL...         -> PASSED
----------------------------------------------------------------------
Ran 7 tests in ~14.5s (100% PASSED, 0 FAILURES)
```

---

## 7. Repository Structure

```
NeuralScan/
├── ai/                           # Edge-AI Inference Modules
│   ├── background_remover.py     # u2net saliency segmentation
│   └── depth_estimator.py        # Depth Anything V2 (CUDA FP16)
│
├── vision/                       # Computer Vision & Optical Tracking
│   ├── camera_stream.py          # Fixed-exposure frame capture (DirectShow)
│   └── laser_tracker.py          # Analytical subpixel center-of-mass (<1 ms)
│
├── fusion/                       # Metric Scaling & Calibration
│   └── metric_aligner.py         # OLS scale regression & 3D back-projection
│
├── reconstruction/               # 3D Fusion & Mesh Processing
│   ├── cloud_integrator.py       # Multi-view R_y(theta) integration & SOR filter
│   └── mesh_exporter.py          # Screened Poisson & watertight STL repair
│
├── hardware/                     # Embedded Interfaces
│   └── serial_controller.py      # UART communication & settling delay sync
│
├── firmware/                     # Microcontroller Source Code
│   ├── uln2003_stepper_controller/   # Arduino Uno + 28BYJ-48 (Primary)
│   └── l298p_stepper_controller/     # Arduino Mega + L298P (Alternative)
│
├── docs/                         # ASUS Hackathon Submission Materials
│   ├── asus_proposal_deck_20_pages.md # Full 20-page proposal deck in English
│   ├── presentation_guide.md          # Slide presentation structure & styling
│   └── pitch_deck_script.md           # 3.5-minute speaker script & Q&A guide
│
├── output/                       # Generated Scan Assets
│   ├── frames/                   # Captured differential frame pairs
│   └── models/                   # Reconstructed 3D CAD models
│       └── neuralscan_model.stl  # Watertight binary STL for 3D printing
│
├── config.py                     # Optical intrinsics, geometry & parameters
├── main.py                       # CLI orchestrator & benchmark engine
├── test_pipeline.py              # Automated 7-stage test suite
├── requirements.txt              # Pinned Python package dependencies
├── README.md                     # Official documentation (English)
└── README_RU.md                  # Documentation in Russian
```

---

## 8. License & Contest Attribution

Developed for the **ASUS UGen AI LEAGUE 2026** (Track: *Battlefield Lightning Global*, Scenario: *Workplace AI*).  
Distributed under the open-source [MIT License](LICENSE).
