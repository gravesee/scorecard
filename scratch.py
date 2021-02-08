from scorecard.transform import *


v = ContinuousTransform([-3, -2, -1, 0, 1, 2, 3], [-998], np.nan)

v.labels
v.expand(0, -5)
v.collapse([0, 1])

x = np.random.randn(10000)

pd.Series(v.to_index(x)).value_counts()

v.labels
v.collapse([0, 2])

v.labels

v.to_categorical(x).value_counts(sort=False)


v = CategoricalTransform(list("abcde"), ["Z"], -998)

v.collapse([2, 3])
v.levels

v.expand(2)

x = np.random.choice(list("abdcde"), size=10000, replace=True)
v.to_index(pd.Series(x))
v.to_categorical(pd.Series(x))
v.to_sparse(pd.Series(x))