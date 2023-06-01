import torch
import numpy as np
import matplotlib.pyplot as plt

pics = torch.rand((1, 256, 188, 334))
topk = 10
min_activates = []
max_activates = []


outputs = []

