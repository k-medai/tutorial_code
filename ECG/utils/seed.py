import os
import numpy as np
import torch

def seed(seed_num) :
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed_num)
    
    np.random.seed(seed_num)
    
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only= True)