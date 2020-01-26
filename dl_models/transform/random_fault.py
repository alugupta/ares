import numpy as np
import random
import torch

from dl_models.transform.rng_subsys import rng_subsys

from dl_models.transform import ModelTransform

class RandomFault(ModelTransform):
  def __init__(self, layer_mask=None, seed=0, frac=0, random_addrs=False, fault_type="uniform", int_bits=2, frac_bits=6):
    super(RandomFault,self).__init__(layer_mask)
    self.frac         = frac
    self.random_addrs = random_addrs
    self.random_seed  = seed
    self.fault_type   = fault_type
    self.int_bits = int_bits
    self.frac_bits = frac_bits
    self.total_bits = frac_bits + int_bits

    self.rng_subsys = rng_subsys('exact', int_bits, frac_bits)

  def __call__(self, model):
    def fault_wrapper(w):

        def quantize(q, v):
          (qi, qf)     = q
          (imin, imax) = (-np.exp2(qi-1), np.exp2(qi-1)-1)
          fdiv         = (np.exp2(-qf))
          v.div_(fdiv).round_().mul_(fdiv)
          v.clamp_(min=imin, max=imax)

        def bit_inject(output, thres, n_bits): 
          self.rng_subsys.start_gen(output, thres)
          qi, qf = n_bits 
          
          sign = (torch.sign(output)*2 + 1).sign()      #Grab the sign of each entry to reimpose after bit flip, by taking {-1,0,1}->{-1,1,2}->{-1,1}          

          for b in range(0,qi-1,1): #one int bit used for sign
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
          
          for b in range(-1,-1*qf - 1, -1): 
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
            uniform, nonzero = self.rng_subsys.generate()
            if nonzero:
                  sign = (output.sign() *2 + 1).sign()
                  fault = torch.where(uniform == 1, torch.empty(1, device=output.device).fill_(1 + 2 ** (self.int_bits-1)), torch.zeros(1, device=output.device))
                  fault *= sign
                  #In 2s comp (assumed for weights), sign bit flipping is equivalent to adding/subtracting the max representable val 
                  output.sub_(fault)
          

        def _inject(w):   #CONVERTED TO HERE
            addrs       = list(range(len(w)))
            if self.random_addrs:
              np.random.shuffle(addrs)

            num_faults = int(len(w) * self.total_bits * self.frac)
            #print("There are %d weights /n", len(w))
            #print("There will be %d bit flips /n", num_faults)
            # Generating random values with np.random (vectorized) is must faster
            # than python random.random
            faults = None
            if self.fault_type == "uniform":
              min_w   = np.min(w)
              max_w   = np.max(w)
              faults = np.random.uniform(min_w, max_w, num_faults)
            elif self.fault_type == "normal":
              mean, sigma = np.mean(w), np.std(w)
              faults = np.random.normal(mean, sigma, num_faults)
            elif self.fault_type == "sign":
              # -1 means flip sign, 1 means maintain sign.
              # 50% chance of flipping sign
              faults = np.random.choice([-1, 1], num_faults)
              for i in range(num_faults):
                faults[i] = faults[i] * w[i]
            elif self.fault_type == "percent":
              #-1 means increase by percent, 1 means decrease by percent
              #just set at 10% changes for now
              percent = 0.1
              faults = np.random.choice([-1,1], num_faults)
              for i in range(num_faults):
                faults[i] = w[i] * (1 + (faults[i] * percent))
            elif self.fault_type == "bit":
              # Eventually we should make sure we're not hitting the same bit.
              #  fine for now though
              bit_inject(w, self.frac, (self.int_bits, self.frac_bits))
            else:
              assert False, "Fault type: %s is invalid" % self.fault_type

            if self.fault_type == "bit":
              pass
              #print("Already updated.")
            else:
              if num_faults > 0:
                fault_addrs = addrs[:num_faults]
                for i in range(num_faults):
                  w[i] = faults[i]

            return w

      ########################################
        size = w.size()
        w = w.flatten()
        return _inject(w).view(size)
    self.transform_weights(model, fault_wrapper)
    
      ########################################

    ##########################################################################
  

