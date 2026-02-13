from operads import *

BE2 = BarrattEccles(2)
x = BE2([[]])
BarrattEccles.compose(x, 1, x)
