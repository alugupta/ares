import numpy as np

import operator as op
import sys
import random
import math
import torch

class reluClip:
    def __init__(self, maxv):
        self.maxv = maxv

    def injection(self, module, input, output):
        output.clip_(maxv=self.maxv)
