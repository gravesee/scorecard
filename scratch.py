from scorecard.transform import *
import numpy as np
from scorecard.performance import BinaryPerformance


y = np.random.choice([0, 1], size=10000, replace=True)
w = np.random.random(10000)




v = ContinuousTransform([-3, -2, -1, 0, 1, 2, 3], [-998, -999], np.nan)

v.labels
v.expand(0, -5)
v.collapse([0, 1])

x = np.random.randn(10000)

pd.Series(v.to_index(x)).value_counts()



v.labels
v.collapse([0, 2])



v.labels

s = v.to_categorical(x)

perf = BinaryPerformance(y, w)
print(perf.summarize(s))

quit()



v = CategoricalTransform(list("abcde"), ["Z"], -998)

v.collapse([2, 3])
v.levels

v.expand(2)

x = np.random.choice(list("abdcde"), size=10000, replace=True)
v.to_index(pd.Series(x))
v.to_categorical(pd.Series(x))
v.to_sparse(pd.Series(x))