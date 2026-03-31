# Maya-Manas — Paper 7: Oscillatory Thalamo-Cortical Gating for Continual Learning in SNNs

**Venkatesh Swaminathan | Nexus Learning Labs, Bengaluru**
**ORCID: 0000-0002-3315-7907**

Part of the Maya Research Series — grounding spiking neural network architecture in Advaita Vedantic cognitive philosophy.

---

## What is Manas?

Manas (मनस्) is the oscillating, doubting, sensory-receiving mind — the first component of the Antahkarana in Advaita Vedanta. It does not decide. It oscillates and presents.

Computationally, Manas maps to thalamo-cortical oscillatory gating. The membrane threshold of the fc1 LIF layer descends monotonically across T_STEPS timesteps, tracing the Vikalpa-to-Sankalpa transition within each forward pass:
```
V_threshold(t) = V_base + A_manas * cos(pi * t / (T_STEPS - 1))

t=0: V_base + A_manas  — Vikalpa (maximum suppression)
t=3: V_base - A_manas  — Sankalpa (full receptivity)
```

Only signals salient enough to spike at high threshold (early timesteps) qualify for amplified Vairagya protection via the Manas-GANE intersection. Noise spikes riding the open window do not.

---

## Results — Split-CIFAR-100 CIL (seed=42)

| Condition | AA | BWT |
|---|---|---|
| A: Maya-Chitta baseline (A_manas=0) | 14.35% | -52.68% |
| B: Manas structure, static threshold | 14.35% | -52.68% |
| C: Manas high amplitude (A_manas=0.25) | 13.01% | -53.00% |
| D: Manas low amplitude (A_manas=0.05) | 15.08% | -51.87% |
| E: Full Maya-Manas (A_manas=0.10) | **15.19%** | **-50.91%** |

**+0.84pp AA and +1.77pp BWT over P6 baseline.**
Best forgetting score in the Maya series to date.

---

## Series Constants Confirmed

- **Bhaya Quiescence Law** — Bhaya fires 0.000 throughout under replay. Confirmed P1-P7. Now a named series law.
- **Buddhi S-curve determinism** — identical across all ablation conditions. Confirmed P4-P7.
- **Manas negative control** — structure alone (B) = baseline (A) exactly. The oscillation is the mechanism, not the wiring.

---

## Repository Structure
```
Maya-Manas/
├── run_manas_cil.py          # Main experiment runner
├── run_ablation_manas.py     # 5-condition ablation
├── sign_paper.py             # IP protection — LSB steganographic signing
├── maya_cl/
│   ├── network/
│   │   ├── backbone.py       # MayaManasNet — O-LIF fc1 layer
│   │   └── affective_state.py
│   ├── plasticity/
│   │   ├── manas.py          # ManasGate + ManasConsistency (P7 core)
│   │   ├── chitta.py         # Carried forward from P6
│   │   ├── viveka.py         # Carried forward from P5
│   │   └── vairagya_decay.py
│   └── utils/
│       └── config.py         # All hyperparameters
└── results/                  # Full ablation CSVs
```

---

## Maya Research Series

| Paper | Mechanism | Benchmark | AA |
|---|---|---|---|
| P1 | Nociceptive Metaplasticity | — | — |
| P2 | Maya-OS Arbitration | — | — |
| P3 | Maya-CL: Bhaya+Vairagya+Shraddha+Spanda | Split-CIFAR-10 TIL | 62.38% |
| P4 | Maya-Smriti: Buddhi | Split-CIFAR-100 CIL | 31.84% |
| P5 | Maya-Viveka: Viveka+GANE | Split-CIFAR-100 CIL | 16.03% |
| P6 | Maya-Chitta: Chitta+Samskara+Moha | Split-CIFAR-100 CIL | 14.42% |
| P7 | Maya-Manas: O-LIF Oscillatory Gate | Split-CIFAR-100 CIL | **15.19%** |

---

## Hardware & Software

- Python 3.11.9 | PyTorch 2.5.1+cu121 | SpikingJelly 0.0.0.0.14
- NVIDIA RTX 4060 8GB | Windows 11
- Seed: 42 | T_STEPS: 4 | A_manas: 0.10

---

## IP Protection

Every figure signed with LSB steganography via `sign_paper.py`.
ORCID magic number embedded in config: `VAIRAGYA_DECAY_RATE = 0.002315`
Canary: `MayaNexusVS2026NLL_Bengaluru_Narasimha`

---

*Nexus Learning Labs, Bengaluru | 2026*
