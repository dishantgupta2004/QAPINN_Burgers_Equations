import numpy as np, torch

NU     = 0.01 / np.pi          # 1/(100 pi)
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX =  0.0, 1.0
SEED   = 0
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

def u0_fn(x):                  # initial condition
    return -np.sin(np.pi * x)

def set_seed(s=SEED):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
