import numpy as np
import pandas as pd
import math
import statistics

s=[1,2,3,4,5,6]
print(np.std(s))
average=np.mean(s)
_std=math.sqrt(sum([(x - 0) ** 2 for x in s]) / len(s))

print(statistics.pstdev(s))
print(_std)
