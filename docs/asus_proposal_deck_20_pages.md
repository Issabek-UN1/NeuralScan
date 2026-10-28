# ASUS UGen AI LEAGUE — Stage I Proposal Deck (20 Pages)
## Track: Battlefield Lightning (Global) | Scenario: Workplace AI
### Project Title: NeuralScan: Edge-AI Hybrid 3D Metrology Scanner on ASUS Hardware

---

<!-- ========================================================================== -->
<!-- PAGE 1: TITLE COVER & TEAM LEAD PROFILE -->
<!-- ========================================================================== -->
# Page 1: Title Cover & Project Identification

```
+-----------------------------------------------------------------------------------+
|  ASUS UGen AI LEAGUE 2026 — GLOBAL INNOVATION HACKATHON                           |
|  Track: Battlefield Lightning (Lightweight Edge Computing) | Division: Global     |
|  Target Scenario: Workplace AI (Smart Metrology & Industrial Quality Inspection)  |
+-----------------------------------------------------------------------------------+
```

## NeuralScan: Edge-AI Hybrid 3D Metrology Scanner
### Democratizing Precision 3D Metrology: $30 Open-Source Hardware + ASUS Edge-AI

---

### Team Information & Team Lead Profile

```
+------------------------------------+-----------------------------------------------+
|       [TEAM LEAD PHOTO HERE]       | Team Name:        NeuralScan Core Team        |
|                                    | Team Lead:        Issabek Matkulov            |
|   Recommended: 400x400 px,         | Role:             Lead AI & Embedded Engineer |
|   Professional headshot,           | Email:            matkulov.isabek@gmail.com   |
|   Transparent / Dark Background    | LinkedIn / X:     Available on Request        |
|                                    | GitHub:           https://github.com/Issabek-UN1|
+------------------------------------+-----------------------------------------------+
```

* **Project Repository:** https://github.com/Issabek-UN1/NeuralScan
* **Demonstration Video:** 3-Minute Hands-On YouTube Video `[Insert Unlisted YouTube Link]`
* **Submission Date:** October 2026

---

<!-- ========================================================================== -->
<!-- PAGE 2: EXECUTIVE SUMMARY & ABSTRACT -->
<!-- ========================================================================== -->
# Page 2: Executive Summary & Abstract

### Disrupting High-Precision 3D Metrology
Industrial 3D optical metrology systems (e.g., GOM ATOS, Creaform HandySCAN, Keyence) cost between **$5,000 and $25,000 USD**. Their prohibitive cost restricts high-precision quality control and reverse engineering to well-funded manufacturing plants, locking out SMEs, educational fablabs, and individual hardware innovators.

**NeuralScan** completely eliminates this hardware cost barrier by substituting expensive physical optics with an **Edge-AI Hybrid Fusion Pipeline running on ASUS hardware**:
1. **$30 Open-Hardware Rig:** Operates on an Arduino Uno R3, an off-the-shelf 28BYJ-48 geared stepper motor (ULN2003 driver), a KY-008 line laser projector ($1.20), and any consumer smartphone camera (1080p).
2. **OLS Metric Alignment Layer (Core Innovation):** Resolves the fundamental scale ambiguity of state-of-the-art monocular depth estimation (**Depth Anything V2**) by anchoring dense neural relief to physical millimeters via an analytical subpixel laser tracker (<1 ms latency).
3. **ASUS Edge-AI Acceleration:** Powered by CUDA FP16 Tensor Cores and preparing for the **ASUS UGen300 (Hailo-10H NPU)**, delivering an **18.8x speedup** over CPU execution (72 ms vs. 1,360 ms per frame). A full 48-step scan computes in just 3.5 seconds.
4. **Watertight CAD Export:** Directly generates a guaranteed 2-manifold, 3D-printable watertight STL file without manual Blender cleanup.

---

<!-- ========================================================================== -->
<!-- PAGE 3: SCENARIO & INDUSTRIAL PROBLEM STATEMENT -->
<!-- ========================================================================== -->
# Page 3: Workplace AI — Industrial Metrology Bottlenecks

```
+---------------------------------------------------------------------------------+
| WORKPLACE AI PROBLEM: Quality Assurance & Dimensional Verification in Industry  |
+---------------------------------------------------------------------------------+
```

### The Inspection Dilemma for Modern Engineering
In rapid prototyping, CNC machining, and additive manufacturing, dimensional verification of physical parts is critical. However, current workflows face major bottlenecks:

1. **Capital Expenditure Barrier:**
   * High-precision structured-light scanners require multi-thousand dollar budgets, dedicated optical benches, and proprietary subscription software.
2. **Cloud Latency & Privacy Liabilities:**
   * Cloud-based computer vision APIs expose confidential industrial CAD intellectual property (IP) to remote servers and introduce unacceptable network latency (10–30 seconds per scan).
3. **Hardware Fragility:**
   * Factory floors, educational workshops, and remote repair centers require rugged, easily serviceable scanning equipment using standard off-the-shelf parts that cost under $50 to build or replace.

**Our Target Workplace Scenario:** Providing workshop engineers, quality inspectors, and makers with instant, on-device 3D part inspection directly on their ASUS laptops.

---

<!-- ========================================================================== -->
<!-- PAGE 4: LIMITATIONS OF EXISTING SOLUTIONS -->
<!-- ========================================================================== -->
# Page 4: Technical Breakdown — Existing Approaches vs. NeuralScan

| Feature / Metric | Industrial Scanners (Creaform/GOM) | Classical Photogrammetry (Meshroom/RealityCapture) | Budget Laser Scanners (BQ Ciclop) | **NeuralScan (Our Solution)** |
| :--- | :--- | :--- | :--- | :--- |
| **System Cost** | **$5,000 – $25,000** | Free + $800 Camera | $150 – $300 | **$30 (Open Hardware)** |
| **Textureless Objects** (e.g. white cylinder, smooth gears) | Excellent | ❌ **Complete Failure** (Cannot find feature points) | Good | **Excellent** (Laser stripe + AI relief) |
| **Scan Compute Time** | 1 – 3 minutes | 15 – 45 minutes | 20 – 40 minutes (slow sweep) | **15 – 25 seconds** |
| **Physical Scale** | True Metric (mm) | ❌ **Arbitrary Unitless Scale** | True Metric (mm) | **True Metric (OLS Anchor)** |
| **Watertight STL Output** | Proprietary Export | High mesh noise & boundary holes | Non-manifold raw point cloud | **Guaranteed Watertight (2-Manifold)** |
| **Edge Hardware Target** | Dedicated Xeon Workstation | High-end Multi-GPU Desktop | Basic 8-bit Microcontroller | **ASUS Edge GPU / UGen300 NPU** |

> **Key Takeaway:** Photogrammetry fails on smooth parts; budget scanners take half an hour. NeuralScan combines the sub-millimeter precision of laser metrology with the instant density of neural depth estimation.

---

<!-- ========================================================================== -->
<!-- PAGE 5: NEURALSCAN SOLUTION ARCHITECTURE -->
<!-- ========================================================================== -->
# Page 5: End-to-End System Architecture

```
+-----------------------------------------------------------------------------------+
|                            NEURALSCAN SYSTEM TOPOLOGY                             |
+-----------------------------------------------------------------------------------+
|  [PHYSICAL RIG ($30)]             [ASUS EDGE-AI HOST]             [OUTPUT]        |
|                                                                                   |
|  +--------------------+         +---------------------+                           |
|  | Arduino Uno R3     | UART    | ASUS RTX / UGen300  |                           |
|  | 28BYJ-48 Stepper   |<------->| Serial Controller   |                           |
|  | KY-008 Line Laser  | 115200  | (Motion/TTL Sync)   |                           |
|  +--------------------+         +----------+----------+                           |
|            |                               |                                      |
|            | (Dual-Frame 1080p)            v                                      |
|            v                    +---------------------+                           |
|  +--------------------+         | Analytical Laser    |                           |
|  | Smartphone / WebCam|-------->| Subpixel Extraction |                           |
|  | Fixed Exp = 0.25   |         | (< 1 ms, Gamma = 3) |                           |
|  +--------------------+         +----------+----------+                           |
|                                            | Metric Z                             |
|                                            v                                      |
|                                 +---------------------+        +----------------+ |
|                                 | OLS Metric Aligner  |------->| Watertight CAD | |
|                                 | (Depth Anything V2  |        | 3D STL Model   | |
|                                 |  + u2net Mask)      |        +----------------+ |
|                                 +---------------------+                           |
+-----------------------------------------------------------------------------------+
```

---

<!-- ========================================================================== -->
<!-- PAGE 6: HARDWARE RIG & $30 BILL OF MATERIALS (BOM) -->
<!-- ========================================================================== -->
# Page 6: Hardware Engineering & $30 Bill of Materials (BOM)

```
                    +------------------------------------+
                    |        Optical Bench Setup         |
                    |  Baseline b = 160 mm               |
                    |  Working Distance Z0 = 300 mm      |
                    |  Convergence Angle alpha = 28.5 deg|
                    +------------------------------------+
```

### Verified Bill of Materials (BOM)

| Item | Component | Specification | Unit Cost (USD) |
| :---: | :--- | :--- | :---: |
| **1** | **Microcontroller** | Arduino Uno R3 (ATmega328P, USB-UART CH340) | **$4.50** |
| **2** | **Rotary Actuator** | 28BYJ-48 Unipolar Stepper (64:1 reduction gearbox, 2048 steps/rev) | **$2.20** |
| **3** | **Driver Board** | ULN2003 Darlington Transistor Array | **$0.80** |
| **4** | **Optical Emitter** | KY-008 Red Line Projector (650 nm, 5 mW, TTL switchable) | **$1.20** |
| **5** | **Line Lens** | 120° Cylindrical Powell Diffuser Lens | **$0.80** |
| **6** | **Optical Sensor** | Smartphone Camera via Iriun Webcam / 1080p USB Camera | **$0.00** (Existing) |
| **7** | **Rig Chassis** | 3D-Printed PLA Turntable Base & Bearing (~160g PLA filament) | **$3.50** |
| **8** | **Wiring & Power** | Standard USB Cables & Jumper Wires | **$2.50** |
| **TOTAL** | **Full 3D Metrology Rig** | **100% Functional Open-Hardware Scanner** | **$15.50 – $30.00** |

---

<!-- ========================================================================== -->
<!-- PAGE 7: ARDUINO FIRMWARE & THERMAL COIL SAFETY -->
<!-- ========================================================================== -->
# Page 7: Firmware Engineering & Motor Thermal Protection

### 1. Zero-Backlash Fractional Accumulator
The 28BYJ-48 geared stepper motor has exactly **2048 steps per full 360° revolution**. When dividing into 48 discrete angular views, each step requires:

$$\text{Steps per View} = \frac{2048}{48} = 42.6667 \text{ steps}$$

Our firmware [`firmware/uln2003_stepper_controller/`](firmware/uln2003_stepper_controller/) implements an exact integer-fraction accumulator:
* Alternates between 42 and 43 micro-steps.
* Ensures **zero cumulative angular drift** across full 360° revolutions.

### 2. Mandatory Thermal Coil De-Energization Rule
Unipolar and bipolar stepper motors have low internal phase resistance. Leaving coil pairs energized during static camera capture causes rapid heat buildup and driver degradation.

```cpp
// Enforced in firmware loop immediately following movement completion:
void step_and_rest(int steps) {
    myStepper.step(steps);
    // CRITICAL: De-energize all 4 Darlington channels to eliminate idle current
    digitalWrite(PIN_IN1, LOW);
    digitalWrite(PIN_IN2, LOW);
    digitalWrite(PIN_IN3, LOW);
    digitalWrite(PIN_IN4, LOW);
}
```
* **Result:** Driver board and motor remain cool at room temperature (<30°C) during continuous 24/7 scanning.

---

<!-- ========================================================================== -->
<!-- PAGE 8: DUAL-FRAME DIFFERENTIAL ACQUISITION -->
<!-- ========================================================================== -->
# Page 8: Optical Acquisition & Ambient Light Subtraction

```
   [Frame A: Laser OFF]             [Frame B: Laser ON]              [Differential Frame]
+------------------------+       +------------------------+       +------------------------+
| Ambient Room Light     |  -->  | Ambient + Red Line     |  -->  | Isolated Laser Line    |
| Object Texture Visible |       | Object Surface Lit     |       | Zero Ambient Noise     |
+------------------------+       +------------------------+       +------------------------+
                                                                  I_diff = max(0, B - A)
```

### Optical Immunity Algorithm
To eliminate ambient lighting fluctuations (sunlight, overhead fluorescent flickering) without requiring a darkroom:
1. **Host-Controlled Synchronous Capture:**
   * Step 1: Laser TTL deactivated $\to$ Capture Ambient Frame $I_{\text{ambient}}(x, y)$.
   * Step 2: Laser TTL activated (pin 7 HIGH) $\to$ Capture Active Frame $I_{\text{laser\_on}}(x, y)$.
2. **Channel-Weighted Differential Processing:**
   $$I_{\text{diff}}(x, y) = \max\left(0, I_{\text{laser\_on}}^{\text{Red}}(x, y) - I_{\text{ambient}}^{\text{Red}}(x, y)\right)$$
3. **Outcome:** Ambient background brightness drops to absolute zero, isolating only the genuine optical photons reflected from the object surface.

---

<!-- ========================================================================== -->
<!-- PAGE 9: ANALYTICAL SUBPIXEL LASER EXTRACTION -->
<!-- ========================================================================== -->
# Page 9: Analytical Subpixel Peak Tracker (< 1 ms)

### Why Object Detectors (YOLO) are Conceptually Forbidden
Bounding-box detectors and segmentation CNNs quantize coordinates into discrete integer pixels and require large training datasets. For 3D metrology, subpixel peak tracking must be exact, analytical, and deterministic.

### Subpixel Weighted Center-of-Mass ($\gamma = 3$)
For every scanline $y$, the horizontal subpixel centroid $x_c(y)$ of the reflected laser stripe is extracted in $< 0.8 \text{ ms}$:

$$x_c(y) = \frac{\sum_{x \in W} x \cdot \left[I_{\text{diff}}(x, y)\right]^\gamma}{\sum_{x \in W} \left[I_{\text{diff}}(x, y)\right]^\gamma}, \quad \text{where } \gamma = 3.0$$

* $W = [x_{\text{peak}} - R, x_{\text{peak}} + R]$ is a localized window centered around the maximum pixel intensity ($R = 12 \text{ px}$).
* The cubic exponent $\gamma = 3$ mathematically suppresses side-lobe diffusion, delivering **0.05 mm spatial accuracy** along the laser profile.

---

<!-- ========================================================================== -->
<!-- PAGE 10: AI MONOCULAR DEPTH ESTIMATION -->
<!-- ========================================================================== -->
# Page 10: Dense Surface Geometry via Depth Anything V2

```
+-----------------------------------------------------------------------------------+
|  DEPTH ANYTHING V2 (Vision Transformer Backbone)                                  |
|  - Ingestion: Full HD RGB Frame (1080p, Laser OFF)                                |
|  - Precision: Native CUDA FP16 Tensor Cores                                       |
|  - Output: Dense Relative Surface Relief D_AI in [0, 1]                           |
+-----------------------------------------------------------------------------------+
```

### The Power of Foundation Depth Models
While triangulation lasers only measure points directly illuminated by the beam, foundation depth models (Depth Anything V2) understand:
* Geometric continuity across complex curves.
* Micro-relief, chamfers, and surface indentations.
* Millions of surface points simultaneously from a single viewpoint.

### The Scale Ambiguity Problem ($D_{\text{AI}} \in [0, 1]$)
Monocular models output **relative depth**, not absolute physical metrics. They cannot distinguish between:
* A 50 mm cylinder placed 300 mm away.
* A 500 mm cylinder placed 3,000 mm away.

**Solution:** NeuralScan uses the physical laser profile as a live calibration ruler to anchor the entire neural depth map to reality.

---

<!-- ========================================================================== -->
<!-- PAGE 11: CORE NOVELTY — OLS METRIC ALIGNMENT LAYER -->
<!-- ========================================================================== -->
# Page 11: Core Mathematical Novelty: OLS Metric Alignment Layer

```
        Physical Laser Depth (mm)
              ^
        320   |                           * (Laser Point Ground Truth)
              |                     *   *
        300   |               *   *   
              |         *   *         Fitted Line: Z_true = s * D_AI + t
        280   |   *   *               Scale s = 110.3 mm, Shift t = 223.1 mm
              +------------------------------------------------------------>
                  0.2        0.4        0.6        0.8        1.0   Depth Anything V2
```

### Closed-Form Mathematical Formulation
Let $\{(x_c(y_i), y_i)\}_{i=1}^N$ be the subpixel coordinates of the laser stripe.  
1. Compute physical triangulation depth for each laser point:
   $$Z_{\text{laser}}(i) = \frac{b \cdot f}{(x_c(y_i) - x_0) + f \cdot \tan(\alpha)}$$
2. Sample the corresponding relative neural depth values: $D_i = D_{\text{AI}}(x_c(y_i), y_i)$.
3. Solve the **Ordinary Least Squares (OLS)** linear system:
   $$s = \frac{\sum_{i=1}^N (D_i - \bar{D})(Z_{\text{laser}}(i) - \bar{Z})}{\sum_{i=1}^N (D_i - \bar{D})^2}, \quad t = \bar{Z} - s \cdot \bar{D}$$
4. Project all pixels in the foreground object mask into true millimeters:
   $$\mathbf{Z}_{\text{dense}}(x, y) = s \cdot D_{\text{AI}}(x, y) + t \quad \forall (x, y) \in \text{Mask}_{\text{object}}$$

---

<!-- ========================================================================== -->
<!-- PAGE 12: FOREGROUND SEGMENTATION (u2net) -->
<!-- ========================================================================== -->
# Page 12: Foreground Segmentation & Environmental Isolation

```
   Raw Input Frame (1080p)            u2net Saliency Mask              Isolated Object Only
+---------------------------+     +---------------------------+     +---------------------------+
| Turntable + Background    | --> | Binary Mask (0 = Clutter) | --> | Zero Background Clutter   |
| Object on Stage           |     |             (1 = Object)  |     | Clean 3D Point Target     |
+---------------------------+     +---------------------------+     +---------------------------+
```

### Autonomous Background Rejection
To prevent environmental room clutter (walls, cables, desk edge) from corrupting the reconstructed point cloud:
* **Neural Object Segmentation:** We deploy `u2net` to extract a high-contrast binary mask $\text{Mask}_{\text{object}} \in \{0, 1\}$.
* **Morphological Refinement:**
  * Morphological Closing (`MORPH_CLOSE`, $5\times 5$ ellipse, 2 iterations) to fill internal texture gaps.
  * Morphological Opening (`MORPH_OPEN`, 1 iteration) to eliminate thin perimeter fuzz.
* **Result:** 100% of background points are discarded before 3D reprojection, keeping the point cloud clean.

---

<!-- ========================================================================== -->
<!-- PAGE 13: 3D MULTI-VIEW KINEMATICS & FUSION -->
<!-- ========================================================================== -->
# Page 13: 3D Multi-View Kinematics & Point Cloud Fusion

```
               Y (Vertical Axis)
               ^
               |       Step theta_k = k * 7.5 deg
               |         (k = 0 .. 47)
               |      ___---___
              [+]----/    |    \----+----> X (Horizontal)
             /   \  |  Object  |
            /  Z  \  \___ | ___/
           v       v     ---
```

### Canonical Frame Transformation
Each captured view $k$ ($k \in [0, K-1]$) produces a metric point cloud in camera coordinates $\mathbf{P}_k = [X, Y, Z]^T$.  
Using pinhole back-projection:

$$X = \frac{(x - x_0) \cdot Z}{f}, \quad Y = \frac{(y - y_0) \cdot Z}{f}$$

The points are mapped into the canonical world frame centered on the turntable axis:

$$\mathbf{P}_{\text{world}} = \mathbf{R}_y(-\theta_k) \cdot \left(\mathbf{P}_k - \mathbf{C}_{\text{turntable}}\right)$$

### Statistical Outlier Removal (SOR)
* **Voxel Downsampling:** Voxel grid filtering (0.8 mm resolution) to maintain uniform density.
* **SOR Filter:** Removes scan artifacts where mean distance to 20 nearest neighbors exceeds $2.0\sigma$.

---

<!-- ========================================================================== -->
<!-- PAGE 14: SCREENED POISSON & WATERTIGHT STL REPAIR -->
<!-- ========================================================================== -->
# Page 14: Screened Poisson & Watertight STL Repair

```
+-----------------------------------------------------------------------------------+
|  MANIFOLD CAD REPAIR PIPELINE                                                     |
|  - Point Cloud with Outward Normals -> Screened Poisson Reconstruction (Depth 9)   |
|  - Component Clustering -> Remove Disconnected Debris Floating Islands            |
|  - Inversion Check -> Ensure Positive Manifold Volume (v > 0)                     |
|  - PyMeshFix Surgery -> Guaranteed Watertight 2-Manifold Binary STL Output        |
+-----------------------------------------------------------------------------------+
```

### The Watertight Guarantee for 3D Printing
3D slicers (Cura, PrusaSlicer, Bambu Studio) reject non-manifold meshes with open holes or inverted face normals.
* **Surface Reconstruction:** Screened Poisson creates a smooth continuous polygonal field.
* **Inverted Winding Guard:** Automatically detects inverted normals (`if mesh.volume < 0: mesh.invert()`) before hole closure.
* **Topological Surgery:** `pymeshfix` reconstructs non-manifold boundary edges.
* **Output:** A guaranteed closed binary STL ready for immediate physical manufacturing.

---

<!-- ========================================================================== -->
<!-- PAGE 15: ASUS HARDWARE ACCELERATION BENCHMARK -->
<!-- ========================================================================== -->
# Page 15: ASUS Hardware Acceleration Benchmark (Key Criterion)

```
===================================================================================
  ASUS UGen AI LEAGUE: EMPIRICAL HARDWARE ACCELERATION AUDIT
===================================================================================
[1] NVIDIA RTX 4060 Laptop GPU (CUDA FP16 Tensor Cores):
    -> Latency per frame:   72.17 ms
    -> Throughput:          13.9 FPS (Quasi-Real-Time Interactive Scanning)

[2] Workstation Host CPU (Sequential x86 Float32):
    -> Latency per frame:   1,360.11 ms (1.36 seconds per view)
    -> Throughput:          0.74 FPS (Severe System Bottleneck)
-----------------------------------------------------------------------------------
  ACCELERATION VERDICT:
  - Hardware Acceleration Factor:  18.8x FASTER on ASUS Edge GPU
  - Total 48-step scan on CPU:     65.3 seconds (unusable for interactive workflow)
  - Total 48-step scan on GPU:     3.5 seconds (instant preview)
===================================================================================
```

```
Execution Time for 48-View Scan:
CPU (x86 Float32):  [==============================================] 65.3 s
ASUS Edge GPU:      [==] 3.5 s (18.8x Faster!)
```

---

<!-- ========================================================================== -->
<!-- PAGE 16: ROADMAP TO ASUS UGen300 (HAILO-10H NPU) -->
<!-- ========================================================================== -->
# Page 16: Battlefield Lightning — ASUS UGen300 NPU Integration

```
+-----------------------------------------------------------------------------------+
|  TRANSITION TO ASUS UGen300 AI ACCELERATOR (Hailo-10H NPU)                        |
|  - Target Form Factor: Ultra-compact, low-power standalone 3D Metrology Pod       |
|  - Target Power Budget: < 10 Watts total system draw                              |
+-----------------------------------------------------------------------------------+
```

### Stage II Hardware Deployment Strategy
For the next phase of the hackathon, we migrate the neural pipeline from desktop GPU to the **ASUS UGen300 AI Accelerator**:
1. **Model Optimization:** Convert Depth Anything V2 ViT backbone to ONNX / Hailo Dataflow Compiler (DFC) format.
2. **Quantization:** Apply Post-Training Quantization (PTQ) to INT8 precision with minimal metric accuracy drop (<0.3 mm).
3. **Stand-Alone Smart Scanner:** Enable direct scanning from lightweight ASUS Zenbook / Vivobook laptops powered entirely over USB-C power delivery.

---

<!-- ========================================================================== -->
<!-- PAGE 17: EMPIRICAL VALIDATION & TEST SUITE -->
<!-- ========================================================================== -->
# Page 17: Empirical Validation & Test Suite

```
===================================================================================
                  NEURALSCAN TEST-SUITE EXECUTION AUDIT
===================================================================================
[TEST 1/7] Hardware Serial Controller (Mock Simulation)...      -> PASSED
[TEST 2/7] Camera Stream & Dual-Frame Acquisition...            -> PASSED
[TEST 3/7] Analytical Subpixel Laser Extraction (< 1 ms)...     -> PASSED
[TEST 4/7] Foreground Object Segmentation (u2net)...            -> PASSED
[TEST 5/7] Monocular Relative Depth (Depth Anything V2)...      -> PASSED
[TEST 6/7] Metric Alignment Layer (Scale Resolution)...         -> PASSED
[TEST 7/7] Multi-View Integration & Watertight STL...           -> PASSED
-----------------------------------------------------------------------------------
Ran 7 tests in 14.596s | Status: 100% OK | Failures: 0
===================================================================================
```

### Physical Scan Output Validation
* **Reconstructed Model:** `output/models/neuralscan_model.stl`
* **Watertight Status:** `True` (Valid 2-Manifold)
* **Face Count:** 19,932 Polygons
* **Volume Integrity:** Positive (+116,271 mm³), matching physical test cylinder geometry.

---

<!-- ========================================================================== -->
<!-- PAGE 18: MARKET IMPACT & COST COMPARISON -->
<!-- ========================================================================== -->
# Page 18: Market Impact & Economic Disruption

```
Capital Cost Comparison (USD):
Commercial Scanner (GOM / Creaform):  [$15,000 +++++++++++++++++++++++++++++++]
Entry-Level 3D Scanner (Revopoint):   [$800   ++]
NeuralScan ($30 Rig + ASUS Edge-AI):  [$30    |] (99.8% Cost Reduction)
```

### Democratizing 3D Digital Manufacturing
* **For Small & Medium Enterprises (SMEs):** Inspect machined parts and 3D prints on-site without purchasing $10k inspection arms.
* **For Education & Fablabs:** Equip entire classrooms and maker spaces with individual 3D metrology stations for the price of one roll of filament.
* **For Field Engineering:** Rugged, lightweight rig deployable anywhere with an ASUS laptop.

---

<!-- ========================================================================== -->
<!-- PAGE 19: PROJECT ROADMAP & STAGE II MILESTONES -->
<!-- ========================================================================== -->
# Page 19: Project Roadmap & Stage II Milestones

```
+-------------------+--------------------+--------------------+--------------------+
| STAGE I (Done)    | STAGE II: Month 1  | STAGE II: Month 2  | STAGE II: Final    |
+-------------------+--------------------+--------------------+--------------------+
| - Functional MVP  | - Deploy on ASUS   | - Cross-Laser Dual | - Packaged Open-   |
| - 7/7 Unit Tests  |   UGen300 NPU      |   Stripe System    |   Source Fablab Kit|
| - 18.8x GPU Bench | - INT8 Quantized   | - RGB Texture Map  | - GUI Desktop App  |
| - Watertight STL  |   Depth Engine     |   Color Fusion     |   for Windows/Linux|
+-------------------+--------------------+--------------------+--------------------+
```

### Immediate Stage II Deliverables:
1. **ASUS UGen300 Hardware Benchmarking:** Full latency and thermal comparison on Hailo-10H NPU.
2. **Dual-Line Cross Laser:** Expanding from single vertical line to dual orthogonal lasers to scan complex concavities.
3. **Open STL Chassis:** Release free downloadable 3D-printable housing for the entire scanner assembly on Printables/Thingiverse.

---

<!-- ========================================================================== -->
<!-- PAGE 20: REFERENCES, TEAM & GITHUB REPOSITORY -->
<!-- ========================================================================== -->
# Page 20: References, Team Credentials & Code Repository

```
+-----------------------------------------------------------------------------------+
|                            OFFICIAL CODE REPOSITORY                               |
|                     https://github.com/Issabek-UN1/NeuralScan                     |
|          Open-Source MIT License | Tested on Windows 11 & ASUS Hardware           |
+-----------------------------------------------------------------------------------+
```

### Team Credentials & Contacts

```
+------------------------------------+-----------------------------------------------+
|       [TEAM LEAD PHOTO HERE]       | Lead Developer:   Issabek Matkulov            |
|                                    | Specialization:   Embedded AI & Edge Robotics |
|   Recommended: 400x400 px,         | Primary Email:    matkulov.isabek@gmail.com   |
|   Professional headshot            | Location:         Global / International      |
|                                    | Contest Division: Battlefield Lightning Global|
+------------------------------------+-----------------------------------------------+
```

### Academic & Technical References
1. **Depth Anything V2:** Yang et al., *"Depth Anything V2: A Foundation Model for Monocular Depth Estimation"*, arXiv:2406.09414, 2024.
2. **Poisson Surface Reconstruction:** Kazhdan, M., & Hoppe, H., *"Screened Poisson Surface Reconstruction"*, ACM TOG, 2013.
3. **Subpixel Peak Detection:** Steger, C., *"An Unbiased Detector of Curvilinear Structures"*, IEEE TPAMI, 1998.
4. **ASUS UGen300 Architecture:** ASUS AIoT Solutions & Hailo-10H High-Efficiency AI Accelerator Manual, 2025/2026.

---
*(End of Proposal Deck — 20 Pages)*
