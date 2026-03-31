# config.py — Maya-Manas (Paper 7) hyperparameters
# Carries forward P6 (Maya-Chitta) base. Adds Manas O-LIF oscillatory gate.

SEED = 42
T_STEPS = 4
CONV1_CHANNELS = 64
CONV2_CHANNELS = 64
CONV3_CHANNELS = 128
FC1_SIZE = 2048
NUM_CLASSES = 100
TAU_MEMBRANE = 2.0
V_THRESHOLD = 0.3
V_RESET = 0.0
TAU_SHRADDHA = 10.0
TAU_BHAYA = 3.0
TAU_VAIRAGYA = 20.0
TAU_SPANDA = 5.0
TAU_VIVEKA = 50.0
TAU_BUDDHI = 200.0
HEBBIAN_LR = 0.01
LABILITY_INIT = 1.0
LABILITY_PAIN_BOOST = 5.0
LABILITY_DECAY_RATE = 0.95
PAIN_CONFIDENCE_THRESHOLD = 0.25
VAIRAGYA_DECAY_RATE = 0.002315        # ORCID magic number — embedded from P6
VAIRAGYA_PROTECTION_THRESHOLD = 0.3
VAIRAGYA_ACCUMULATE_RATE = 0.0015
VAIRAGYA_PAIN_EROSION_RATE = 0.005
VIVEKA_CONSISTENCY_RISE = 0.01
VIVEKA_CONSISTENCY_DECAY = 0.005
VIVEKA_GAIN_MAX = 3.0
VIVEKA_MIN_TASKS = 2
USE_ORTHOGONAL_HEAD = False
PROTOTYPE_DIM = 2048
NUM_TASKS = 10
CLASSES_PER_TASK = 10
BATCH_SIZE = 128
EPOCHS_PER_TASK = 20
REPLAY_BUFFER_SIZE = 50
REPLAY_RATIO = 0.3
REPLAY_VAIRAGYA_PARTIAL_LIFT = 0.8
REPLAY_PAIN_EXEMPT = True
CIL_BOUNDARY_DECAY = 0.50
CIL_MAX_VFOUT_PROTECTION = 0.70

# Chitta hyperparameters — carried forward from P6, unchanged
CHITTA_SAMSKARA_RISE = 0.002315
CHITTA_SAMSKARA_DECAY = 0.0007
CHITTA_MOHA_THRESHOLD = 0.95
CHITTA_MOHA_RELEASE_RATE = 0.60
CHITTA_MIN_TASKS = 1
CHITTA_GATE_STRENGTH = 0.30

# Manas hyperparameters — Paper 7 new contribution
# O-LIF half-cycle: V_threshold(t) = V_base + A_manas * cos(π * t / (T_STEPS - 1))
# t=0: Vikalpa  — maximum suppression (V_base + A_manas)
# t=3: Sankalpa — full receptivity   (V_base - A_manas)
A_MANAS = 0.10                        # canonical calibrated amplitude
A_MANAS_HIGH = 0.25                   # ablation condition C — spike starvation regime
A_MANAS_LOW = 0.05                    # ablation condition D — partial filtering
MANAS_MIN_TASKS = 0                   # Manas activates from Task 0 — it is perceptual, not memorial

# Manas-GANE intersection threshold
# Only synapses that are BOTH Viveka-consistent AND Manas-peak-aligned
# receive amplified Vairagya protection
MANAS_GANE_PEAK_THRESHOLD = 0.5      # cos(π*t/(T-1)) must exceed this at spike time
                                      # i.e. spike must occur in first half of window
                                      # (closer to Vikalpa→Sankalpa transition)

DATA_DIR = "data/"
RESULTS_DIR = "results/"