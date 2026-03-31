# Maya-Chitta

**Endocannabinoid-Inspired Retrograde Gradient Gating for Class-Incremental Learning in Affective Spiking Neural Networks**

*Maya Research Series · Paper 6 · Nexus Learning Labs, Bengaluru*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19337040.svg)](https://doi.org/10.5281/zenodo.19337040)

**[📊 Interactive Dashboard](https://venky2099.github.io/Maya-Chitta/maya_chitta_dashboard_v2.html)**

---

## Overview

Maya-Chitta introduces the first retrograde gradient mechanism in the Maya Research Series. Drawing on Advaita Vedantic philosophy, we model **Chitta** (चित्त) — the subconscious impression store — as a per-synapse **Samskara** trace system that accumulates cross-task impression history and applies retrograde gradient suppression to prevent **Moha** (over-attachment pathology).

The biological grounding is **endocannabinoid (eCB) retrograde signalling** (Wilson & Nicoll, 2001): post-synaptic neurons releasing retrograde messengers that travel back to pre-synaptic terminals to modulate release probability in proportion to usage history.

---

## Key Results

| Condition | Description | AA (%) | BWT (%) |
|---|---|---|---|
| A · Baseline | SGD, no plasticity, no replay | 6.82 | −62.01 |
| B · Replay Only | Ring buffer, no Maya dimensions | 15.23 | −54.24 |
| C · Maya-Viveka | P5 configuration (comparison baseline) | 14.26 | −52.81 |
| D · Gate Only | Gradient gate, Moha release OFF | 14.29 | −53.04 |
| **E · Maya-Chitta ★** | **Gate + Moha release — canonical** | **14.42** | **−53.12** |
| F · NoGate | Moha release only, gate OFF | 14.42 | −53.12 |

**Total Chitta contribution: +0.16 pp AA over Maya-Viveka**
- Gradient gate alone: +0.03 pp
- Moha boundary release: +0.13 pp
- NoGate = Full (E ≡ F): structural calibration datum — Moha release is dominant at gate strength 0.30

**Benchmark:** Split-CIFAR-100 CIL · 10 tasks · seed=42

---

## Core Mechanism

```python
# Samskara trace update (per-synapse, per step)
S_t = S_{t-1} + α_rise * |w_t| - α_decay * S_{t-1}
# α_rise = 0.002315 (ORCID-derived), α_decay = 0.0007

# Retrograde gradient gate
gate = 1.0 - (traces * active_mask * CHITTA_GATE_STRENGTH)
Δw ← Δw * gate  # CHITTA_GATE_STRENGTH = 0.30

# Moha detection and release (at task boundary)
moha_fraction = mean(S > CHITTA_MOHA_THRESHOLD)  # threshold = 0.95
S[S > 0.95] *= CHITTA_MOHA_RELEASE_RATE           # release rate = 0.60
```

---

## Series Findings

- **Bhaya Quiescence Law** — confirmed for the **fifth consecutive paper**: in any Maya series SNN with a functioning replay buffer, Bhaya firing rate approaches zero. Now formally named.
- **Buddhi S-curve determinism** — identical trajectory across all 6 ablation conditions, independent of all Chitta parameters. Confirmed from P4 through P6.
- **Samskara traces are proactive** — active from Task 0, Epoch 1 (S=0.011), before any cross-task interference has occurred.

---

## Repository Structure

```
Maya-Chitta/
├── maya_cl/
│   ├── plasticity/
│   │   ├── chitta.py           # ChittaSamskara — core P6 contribution
│   │   ├── viveka.py           # Carried from P5
│   │   ├── vairagya_decay.py   # Carried from P3
│   │   └── hebbian.py
│   ├── network/
│   │   ├── affective_state.py  # Extended with chitta signal
│   │   └── lif_layers.py
│   ├── eval/
│   │   ├── logger.py           # samskara_mean, moha_fraction columns
│   │   └── metrics.py
│   └── utils/
│       └── config.py           # All CHITTA_* hyperparameters
├── run_chitta_cil.py            # Main experiment
├── run_ablation_chitta.py       # 6-condition ablation
├── sign_paper.py                # LSB steganographic IP protection
├── maya_chitta_dashboard_v2.html  # Interactive results dashboard
└── results/                     # Ablation CSV logs
```

---

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| CHITTA_SAMSKARA_RISE | 0.002315 | ORCID-derived magic number |
| CHITTA_SAMSKARA_DECAY | 0.0007 | Per-step trace decay |
| CHITTA_GATE_STRENGTH | 0.30 | Gradient suppression strength |
| CHITTA_MOHA_THRESHOLD | 0.95 | Saturation threshold |
| CHITTA_MOHA_RELEASE_RATE | 0.60 | 40% reduction on release |
| VAIRAGYA_DECAY_RATE | 0.002315 | ORCID-derived (carried from P5) |
| SEED | 42 | All runs |

Full hyperparameter table in Appendix A of the paper.

---

## Reproducing Results

```powershell
# Main experiment (Condition E canonical)
python run_chitta_cil.py

# Full 6-condition ablation
python run_ablation_chitta.py
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

| Paper | Title | Benchmark | Key Result | DOI                                                                |
|---|---|---|---|--------------------------------------------------------------------|
| P1 | Nociceptive Metaplasticity | CIFAR-10 TIL | Bhaya + Vairagya | [10.5281/zenodo.19151563](https://doi.org/10.5281/zenodo.19151563) |
| P2 | Maya-OS | OS arbitration | Affective SNN arbitration | [10.5281/zenodo.19160123](https://doi.org/10.5281/zenodo.19160123) |
| P3 | Maya-CL | CIFAR-10 TIL | AA=62.38% | [10.5281/zenodo.19201769](https://doi.org/10.5281/zenodo.19201769) |
| P4 | Maya-Smriti | CIFAR-10 CIL | AA=31.84%, Buddhi | [10.5281/zenodo.19228975](https://doi.org/10.5281/zenodo.19228975) |
| P5 | Maya-Viveka | CIFAR-100 CIL | AA=16.03%, Viveka | [10.5281/zenodo.19279002](https://doi.org/10.5281/zenodo.19279002) |
| **P6** | **Maya-Chitta** | **CIFAR-100 CIL** | **AA=14.42%, Chitta** | [10.5281/zenodo.19337040](https://doi.org/10.5281/zenodo.19337040) |

---

## Citation

```bibtex
@misc{swaminathan2026mayachitta,
  title   = {Maya-Chitta: Endocannabinoid-Inspired Retrograde Gradient Gating
             for Class-Incremental Learning in Affective Spiking Neural Networks},
  author  = {Swaminathan, Venkatesh},
  year    = {2026},
  doi     = {10.5281/zenodo.19337040},
  url     = {https://doi.org/10.5281/zenodo.19337040}
}
```

---

## Author

**Venkatesh Swaminathan**
M.Sc. candidate, Data Science and Artificial Intelligence, BITS Pilani
Nexus Learning Labs, Bengaluru
ORCID: [0000-0002-3315-7907](https://orcid.org/0000-0002-3315-7907)
GitHub: [venky2099](https://github.com/venky2099)