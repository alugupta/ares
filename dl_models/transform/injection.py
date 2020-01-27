import numpy as np

import operator as op
import sys
import random
import math
import torch

from dl_models.transform.rng_subsys import rng_subsys

NUM_REP = "1c"       #Representation for numbers, can be 2s comp or 1s comp

class ActivationFault:  #Inject faults on activations
    def __init__(self, faults, q):
        self.faults = faults
        self.qi = q[0]
        self.qf = q[1]
        self.rng_subsys = rng_subsys('expectation', self.qi, self.qf)

    def injection(self, module, input, output):
      faults = self.faults          
      if faults == None:
        return

      
      if type(output)==tuple:   #If RNN, inject on hidden layers
        output = output[1]
        if type(output)==tuple:   #If LSTM, inject on both cell and hidden
            self.run_inject(output[1].data, faults)
            self.run_inject(output[0].data, faults)
        else: self.run_inject(output.data, faults)
      else:
        self.run_inject(output.data, faults)

    #define helper functions for each type of fault
    def run_inject(self, output, faults):
      #Inject the faults defined in the dictionary faults in the given output
      faults_transformations = faults.keys()

       #Fault injection for sign bits
      if 'sign' in faults_transformations:
        thres = faults['sign']
        self.rng_subsys.start_gen(output, thres)
        self.sign_inject(output, thres, )

      # Bit-level fault injection given a thres and specific bit position
      if 'bit' in faults_transformations:
        thres     = faults['bit'][0]
        qi, qf   = faults['bit'][1]
        self.quantize((qi, qf), output)
        if thres != 0:
            output = output.view(-1)
            self.rng_subsys.start_gen(output, thres)
            self.bit_inject(output, thres)

      # Fault injection for values (uniform distribution)
      if 'uniform' in faults_transformations:
         #Create random number generator
        thres = faults['uniform']
        self.rng_subsys.start_gen(output, thres)
        self.uniform_inject(output, thres)

      # Fault injection for values (normal distribution)
      if 'normal' in faults_transformations:
        thres = faults['normal']
        self.rng_subsys.start_gen(output, thres)
        self.normal_inject(output, thres)

      # Quantizing activations
      if 'quantize' in faults_transformations:
        self.quantize ( faults['quantize'], output )


    def sign_inject(self, output, thres):  
        uniform, nonzero = self.rng_subsys.generate()
        if nonzero:
              if NUM_REP == "2c":
                  sign = (output.sign() *2 + 1).sign()
                  fault = torch.where(uniform == 1, torch.empty(1, device=output.device).fill_(1 + 2 ** (self.qi-1)), torch.zeros(1, device=output.device))
                  fault *= sign
                  #In 2s comp, sign bit flipping is equivalent to adding/subtracting the max representable val 
                  output.sub_(fault)
              else:
                  uniform.sub_(.5).mul_(-2)
                  output.mul_(uniform)

    def bit_inject(self, output, thres): 
      shape = output.shape
      
      sign = (torch.sign(output)*2 + 1).sign()      #Grab the sign of each entry to reimpose after bit flip, by taking {-1,0,1}->{-1,1,2}->{-1,1}

      for b in range(0,self.qi-1,1): #one int bit used for sign
        #generate mask of uniformly distributed random numbers
        uniform, nonzero = self.rng_subsys.generate()
        if nonzero:
            cb = 2 ** b
            # Isolate the value to be modified based on given bit position by shift and mod
            fault = output / cb
            fault.abs_().fmod_(2)
            
            #if the bit was a 1, subtract the value corresponding to that bit to flip it to 0
            #if the bit was a 0, add the value corresponding to that bit to flip it to 1
            fault = torch.where(fault < 1, torch.empty(1, device=output.device).fill_(cb), 
                torch.empty(1, device=output.device).fill_(-cb))
            
            
            #apply flip only to 'thres' fraction of elements using generated mask
            fault *= uniform
            # return output with added flips and appropriate sign
            output += fault * sign

      count=0
      for b in range(-1,-1*self.qf - 1, -1): 
        uniform, nonzero = self.rng_subsys.generate()
        if nonzero:
            cb = 2 ** b
            # Isolate the value to be modified based on given bit position by shift and mod
            fault = output / cb
            fault.abs_().fmod_(2)

            #if the bit was a 1, subtract the value corresponding to that bit to flip it to 0
            #if the bit was a 0, add the value corresponding to that bit to flip it to 1
            fault_sw = torch.where(fault < 1, torch.empty(1, device=output.device).fill_(cb), 
                torch.empty(1, device=output.device).fill_(-cb))

            #apply flip only to 'thres' fraction of elements using generated mask
            fault_sw *= uniform

            # return output with added flips and appropriate sign
            output += fault_sw * sign
      
      #flip sign bit after other bit flips, in case a zero got hit above

      if qi > 0: # is there a sign bit to flip?
        self.sign_inject(output, thres)
                  
      output = output.view(shape)      
        
    def uniform_inject(self, output, thres):  
      uniform, _ = self.rng_subsys.generate()  
      minv  = torch.min(output).item()
      maxv  = torch.max(output).item()

      noise = torch.rand_like(output) * (maxv - minv) + minv
      
      noise = torch.where(uniform < thres, noise, torch.empty(1, device=output.device).fill_(1))
      output.mul_(noise)
      

    def normal_inject(self, output, thres):
      uniform, _ = self.rng_subsys.generate()

      mean, sigma = torch.mean(output), torch.std(output)
      
      noise = (torch.randn_like(output) + mean) * sigma 
      noise = torch.where(uniform < thres, noise, torch.empty(1, device=output.device).fill_(1))

      output.mul_(noise)
      

    def quantize(self, v):
      (imin, imax) = (-np.exp2(self.qi-1), np.exp2(self.qi-1)-1)
      fdiv         = (np.exp2(-self.qf))

      v.div_(fdiv).round_().mul_(fdiv)
      v.clamp_(min=imin, max=imax)



