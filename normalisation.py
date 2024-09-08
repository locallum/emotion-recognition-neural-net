import math
import numpy as np

layer_outputs = [4.8, 1.21, 2.385]

E = math.e

exp_values = np.exp(layer_outputs)

norm_values = exp_values / np.sum(exp_values)

print(norm_values)
print(sum(norm_values))
      

