# Maya-Manas

**Oscillatory Thalamo-Cortical Gating for Class-Incremental Learning in Affective Spiking Neural Networks**

*Maya Research Series · Paper 7 · Nexus Learning Labs, Bengaluru*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

**[📊 Interactive Dashboard](https://venky2099.github.io/Maya-Manas/maya_manas_dashboard.html)**

---

## Overview

Maya-Manas introduces the first oscillatory attentional gate in the Maya Research Series. Drawing on Advaita Vedantic philosophy, we model **Manas** (मनस्) — the oscillating, doubting, sensory-receiving mind — as a per-timestep threshold modulation on the fc1 LIF layer, tracing the **Vikalpa-to-Sankalpa** transition within each forward pass.

The biological grounding is **thalamo-cortical oscillatory gating** (Steriade et al., 1993): thalamic bursts precede cortical receptivity — the gate opens, inputs flow through, then closes after the presentation ends. The full oscillatory cycle spans multiple input presentations, not a single forward pass.

The **Manas-GANE intersection** restricts amplified Vairagya protection to synapses that are both spatially consistent (Viveka-qualified) AND temporally salient (fired during the high-threshold Vikalpa phase) — preventing noise spikes from hijacking GANE amplification.

---

## Key Results

| Condition | Description | AA (%) | BWT (%) |
|---|---|---|---|
| A · Maya-Chitta baseline | P6 configuration, A_manas=0 | 14.35 | −52.68 |
| B · Manas structure | O-LIF wired, A_manas=0, no oscillation | 14.35 | −52.68 |
| C · High amplitude | A_manas=0.25 — spike starvation regime | 13.01 | −53.00 |
| D · Low amplitude | A_manas=0.05 — partial filtering | 15.08 | −51.87 |
| **E · Full Maya-Manas ★** | **A_manas=0.10 — canonical** | **15.19** | **−50.91** |

**Total Manas contribution: +0.84 pp AA, +1.77 pp BWT over P6 baseline**
- Structure alone (B = A): zero contribution — the oscillation is the mechanism, not the wiring
- Best BWT in the Maya series to date: −50.91%

**Benchmark:** Split-CIFAR-100 CIL · 10 tasks · seed=42

---

## Core Mechanism
```python
# O-LIF half-cycle threshold schedule
# Vikalpa (suppression) → Sankalpa (receptivity) across T_STEPS=4
V_threshold(t) = V_base + A_manas * cos(pi * t / (T_STEPS - 1))

# t=0: 0.400  — Vikalpa (maximum suppression)
# t=1: 0.350
# t=2: 0.250
# t=3: 0.200  — Sankalpa (full receptivity)

# Manas-GANE intersection
# Only synapses that are BOTH Viveka-consistent AND Manas-peak-aligned
# receive amplified Vairagya protection
manas_gane_mask = manas_consistency(viveka_scores > 0.3) & peak_active_expanded
vairagya_gain[manas_gane_mask] *= 2.0
```

---

## Series Findings

- **Bhaya Quiescence Law** — confirmed for the **seventh consecutive paper**: in any Maya series SNN with a functioning replay buffer, Bhaya firing rate approaches zero under replay. Now formally named as a series law.
- **Buddhi S-curve determinism** — identical trajectory across all 5 ablation conditions. Confirmed P4 through P7.
- **Manas negative control** — B = A exactly. Structure without oscillation contributes nothing. The gate must open.
- **Amplitude sensitivity** — AA peaks at A_manas=0.10. Too high (C) causes spike starvation. Too low (D) gives partial filtering only.

---

## Repository Structure
```
Maya-Manas/
├── maya_cl/
│   ├── plasticity/
│   │   ├── manas.py            # ManasGate + ManasConsistency — core P7 contribution
│   │   ├── chitta.py           # Carried from P6
│   │   ├── viveka.py           # Carried from P5
│   │   └── vairagya_decay.py   # Carried from P3
│   ├── network/
│   │   ├── backbone.py         # MayaManasNet — O-LIF fc1 layer
│   │   └── affective_state.py  # Extended with manas signal
│   ├── eval/
│   │   ├── logger.py           # manas_peak_fraction column added
│   │   └── metrics.py
│   └── utils/
│       └── config.py           # All A_MANAS_* hyperparameters
├── run_manas_cil.py             # Main experiment
├── run_ablation_manas.py        # 5-condition ablation
├── sign_paper.py                # LSB steganographic IP protection
├── maya_manas_dashboard.html    # Interactive results dashboard
└── results/                     # Ablation CSV logs
```

---

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| A_MANAS | 0.10 | Canonical oscillation amplitude |
| A_MANAS_HIGH | 0.25 | Ablation C — spike starvation |
| A_MANAS_LOW | 0.05 | Ablation D — partial filtering |
| MANAS_GANE_PEAK_THRESHOLD | 0.50 | cos threshold for peak-alignment |
| VAIRAGYA_DECAY_RATE | 0.002315 | ORCID-derived (carried from P5) |
| CHITTA_SAMSKARA_RISE | 0.002315 | ORCID-derived (carried from P6) |
| SEED | 42 | All runs |
| T_STEPS | 4 | Timesteps per forward pass |

---

## Reproducing Results
```powershell
# Main experiment (canonical A_manas=0.10)
python run_manas_cil.py

# Full 5-condition ablation
python run_ablation_manas.py
```

Requirements: Python 3.11.9 · PyTorch 2.5.1+cu121 · SpikingJelly 0.0.0.0.14 · NVIDIA GPU

---

## IP Protection

All figures are steganographically signed via `sign_paper.py` (LSB embedding, ORCID watermark). Canary string logged at every experiment run:
```
MayaNexusVS2026NLL_Bengaluru_Narasimha
```

---

## Maya Research Series

| Paper | Title | Benchmark | Key Result | DOI |
|---|---|---|---|---|
| P1 | Nociceptive Metaplasticity | CIFAR-10 TIL | Bhaya + Vairagya | [10.5281/zenodo.19151563](https://doi.org/10.5281/zenodo.19151563) |
| P2 | Maya-OS | OS arbitration | Affective SNN arbitration | [10.5281/zenodo.19160123](https://doi.org/10.5281/zenodo.19160123) |
| P3 | Maya-CL | CIFAR-10 TIL | AA=62.38% | [10.5281/zenodo.19201769](https://doi.org/10.5281/zenodo.19201769) |
| P4 | Maya-Smriti | CIFAR-10 CIL | AA=31.84%, Buddhi | [10.5281/zenodo.19228975](https://doi.org/10.5281/zenodo.19228975) |
| P5 | Maya-Viveka | CIFAR-100 CIL | AA=16.03%, Viveka | [10.5281/zenodo.19279002](https://doi.org/10.5281/zenodo.19279002) |
| P6 | Maya-Chitta | CIFAR-100 CIL | AA=14.42%, Chitta | [10.5281/zenodo.19337041](https://doi.org/10.5281/zenodo.19337041) |
| **P7** | **Maya-Manas** | **CIFAR-100 CIL** | **AA=15.19%, Manas** | *pending* |

---

## Citation
```bibtex
@misc{swaminathan2026mayamanas,
  title   = {Maya-Manas: Oscillatory Thalamo-Cortical Gating for
             Class-Incremental Learning in Affective Spiking Neural Networks},
  author  = {Swaminathan, Venkatesh},
  year    = {2026},
  note    = {Zenodo preprint, DOI pending}
}
```

---

## Author

**Venkatesh Swaminathan**
M.Sc. candidate, Data Science and Artificial Intelligence, BITS Pilani
Nexus Learning Labs, Bengaluru
ORCID: [0000-0002-3315-7907](https://orcid.org/0000-0002-3315-7907)
GitHub: [venky2099](https://github.com/venky2099)
