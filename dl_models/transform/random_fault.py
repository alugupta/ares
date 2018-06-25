import numpy as np
import random

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

  def __call__(self, model):
    def fault_wrapper(w):
      def convert_b(num_float):
        orig_flt=num_float
        int_bits=self.int_bits
        frac_bits=self.frac_bits
        x=np.zeros(int_bits+frac_bits)
        # figure out where we're starting from
        #  also take care of the first bit.
        neg_num=False
        int_offset=0
        if num_float < 0.:
          neg_num=True
          x[0]=1
        else:
          x[0]=0

        # In doing it this way we can to pos&neg numbers the same way..
        #  i think this is the cleanest way to do it..
        if neg_num:
          current_num=-1.*2**(int_bits-1)
        else:
          current_num=0
        if True:
          xid=1
          for iid in list(reversed(range(int_bits-1))):
            if current_num + 2.0**(iid) <= orig_flt:
              current_num += 2.0**(iid)
              x[xid]=1
            xid+=1
          for fid in range(1, frac_bits+1):
            if current_num + 0.5**(fid) <= orig_flt:
              current_num += 0.5**(fid)
              x[xid]=1
            xid+=1

        if False:
          print orig_flt, x
        return x

      def convert_f(x):
        current=0
        int_bits=self.int_bits
        frac_bits=self.frac_bits
        if x[0] == 1:
          current-=2.0**(int_bits-1)
        xid=1
        for iid in list(reversed(range(int_bits-1))):
          if x[xid]==1:
            current+=2.0**(iid)
          xid+=1
        for fid in range(1, frac_bits+1):
          if x[xid]==1:
            current+=0.5**(fid)
          xid+=1
        if False:
          print current
        return current

      def _inject(w):
        addrs       = range(len(w))

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
          faulty_weights=np.random.randint(0,len(w),num_faults)
          faulty_bits   =np.random.randint(0,self.total_bits,num_faults)

          for fw, fb in zip(faulty_weights, faulty_bits):
            # get the origional weight and convert it to an array of bits
            orig_w = w[fw]
            bit_rep_w = convert_b(orig_w)

            #flip the faulty bit
            if False:
              print 'Orig:', bit_rep_w
            bit_rep_w[fb] = 1 - bit_rep_w[fb]
            if False:
              print 'Now:', bit_rep_w

            # convert the corrupted bit representation back to a float
            #   and rewrite the value
            faulty_w = convert_f(bit_rep_w)
            if False:
              print orig_w, faulty_w
              print '-----------------'
            w[fw]=faulty_w

        else:
          assert False, "Fault type: %s is invalid" % self.fault_type

        if self.fault_type == "bit":
          print "Already updated."
        else:
          if num_faults > 0:
            fault_addrs = addrs[:num_faults]
            for i in range(num_faults):
              w[i] = faults[i]

        return w

      ########################################
      np.random.seed(self.random_seed)
      shape       = w.shape
      w_flattened = w.flatten()
      #print("Shape is ", shape, " length of flattened is ", len(w_flattened))
      w           = _inject(w_flattened).reshape(shape)

      return w
      ########################################

    ##########################################################################
    self.transform_weights(model,fault_wrapper)
