from scorecard.scorecard import Scorecard
from scorecard.transform import *
import numpy as np
from scorecard.performance import BinaryPerformance
from scorecard.model import zip_coefs_and_variables

from scorecard.discretize import discretize
import seaborn as sns

df = sns.load_dataset("titanic")

X = df.drop(columns=["survived"])
perf = BinaryPerformance(df.survived)

# mod = discretize(df.drop(columns=['survived']), perf=perf, max_leaf_nodes=6, min_samples_leaf=50)

mod = Scorecard.discretize(X, perf=perf, max_leaf_nodes=6, min_samples_leaf=50)

mod['pclass'].step = 1
mod['sex'].step = 1
mod['fare'].step = 1

# mod["pclass"].decreasing_constraints()
# mod["pclass"].neutralize(0)
mod["pclass"].set_constraint(0, 2, "=")

mod.fit(alpha=1)

print(mod.display_variable('pclass'))

print(zip_coefs_and_variables(mod.model.coefs, mod.variables))

quit()
print("COEFS", mod.model.coefs)

phat = mod.predict(X)

from sklearn.metrics import roc_auc_score

print(roc_auc_score(df.survived, phat))
print(mod.models)
# print(mod.model.coefs)

mod['sex'].step = None
mod.fit(X, perf, alpha=1)
print(mod.model.coefs)
phat = mod.predict(X)
print(roc_auc_score(df.survived, phat))

print(mod.models)

mod.load_model("model_00")
phat = mod.predict(X)
print(roc_auc_score(df.survived, phat))



quit()

M = mod.to_sparse(step=[None])
from scipy.sparse import hstack

print(hstack([M, np.ones(M.shape[0]).reshape(-1, 1)]).shape)



mod["sex"].increasing_constraints()
print(mod["sex"])

mod["sex"].collapse([0, 1])
print(mod["sex"])

# print(mod['sex'])

mod["sex"].expand(0)
print(mod["sex"])

mod["sex"].clear_constraints()
print(mod["sex"])

# print(mod['sex'])

print(mod["fare"])

quit()
# print(mod['fare'].labels)

z = perf.summarize(mod["sex"].to_categorical(df["sex"]))
print(z)

z = perf.summarize(mod["fare"].to_categorical(df["fare"]))
print(z)


quit()
N = 1000000

y = np.random.choice([0, 1], size=N, replace=True)
w = np.random.random(N)


v = ContinuousTransform([-3, -2, -1, 0, 1, 2, 3], [-998, -999], np.nan)

v.labels
v.expand(0, -5)
v.collapse([0, 1])

x = np.random.randn(N)

pd.Series(v.to_index(x)).value_counts()


v.labels
v.collapse([0, 2])


v.labels

v.collapse([0, 1])
v.expand(0, 2)
s = v.to_categorical(x)

# perf = BinaryPerformance(y, w)

from scorecard.variable import Variable

var = Variable(v)

# %timeit perf.summarize(s, cache=False)

s = v.to_categorical(x)
v.reset()

quit()


v = CategoricalTransform(list("abcde"), ["Z"], -998)

v.collapse([2, 3])
v.levels

v.expand(2)

x = np.random.choice(list("abdcde"), size=10000, replace=True)
v.to_index(pd.Series(x))
v.to_categorical(pd.Series(x))
v.to_sparse(pd.Series(x))
