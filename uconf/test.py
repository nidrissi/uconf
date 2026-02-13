from operads import *

SE2 = Surjection(2)
u = SE2([1, 2])
v = SE2([1, 2])
Surjection.compose(u, 1, v)

BE2 = BarrattEccles(2)
x = BE2(([1, 2], [2, 1], [1, 2]))
print(x)
u = x.table_reduction()
print(u)
