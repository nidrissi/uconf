from operads import *

BE3 = BarrattEccles(3)
x = BE3(([1, 2, 3], [2, 3, 1], [3, 1, 2]))
x.complexity()
y = x.permute([2, 3, 1])
print(y)

BE4 = BarrattEccles(4)
be3_4 = BE3.tensor(BE4)
be3_4.term("foobar")
