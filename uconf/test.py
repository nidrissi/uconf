from operads import *

SE2 = Surjection(2)
u = SE2([1, 2])
v = SE2([1, 2])
Surjection.compose(u, 1, v)
