import numpy as np
import random
import torch


class rng_subsys:           #Handle generation of numbers for weight and activation injections. 
                                #Can generate the exact # of expected errors per layer, or have each bit have 'thres' chance of faulting
    def __init__(self, mode, qi, qf):
        if mode == 'exact': 
            self.generate = self.gen_exact
        if mode == 'expectation':
            self.generate = self.gen_exp
        self.total_bits = qi+qf
        
    def start_gen(self, output, thres):    #Only matters for exact generation
        self.count = 0
        self.shape = output.shape
        self.device = output.device
        self.thres = thres

        if self.generate == self.gen_exact:
            self.zero = torch.zeros(1,device=output.device, dtype=torch.long)
            error_count = int(len(output) * thres * (self.total_bits))

            self.loc = torch.randint(0, len(output), [error_count], device = output.device, dtype=torch.long) #Generate the location and address for each error
            self.bit = torch.randint(0, self.total_bits, [error_count],  device = output.device)

    def gen_exact(self):
        ones = torch.ones(self.shape, device=self.device)
        bit_loc = torch.where(self.bit == self.count, self.loc, self.zero).nonzero().flatten()
        uniform = torch.zeros(self.shape, device=self.device)
        if bit_loc.nelement() > 0:
            uniform.scatter_(0, self.loc[bit_loc], ones)

        self.count += 1
        return (uniform, bit_loc.nelement() > 0)

    def gen_exp(self):
        index = torch.ones(self.shape, device=self.device)
        zero = torch.zeros(1, device=self.device)
        thres = self.thres
        while thres < 1e-5:     #The pytorch CUDA RNG does not generate numbers near 00 with uniform probability.
                                    #To work around this, product several draws where accuracy is better
            uniform = torch.rand(self.shape, device=self.device)
            index = torch.where(uniform < 1e-5, index, zero)
            thres /= 1e-5
        uniform = torch.rand(self.shape, device=self.device)
        index = torch.where(uniform < thres, index, zero)

        self.count += 1
        return (index, index.nonzero().nelement() > 0)
