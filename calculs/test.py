from uconf import *

BBE3 = BarConstruction(BarrattEccles, 3)
x = BBE3(LabelledOrderedTree([], 42))
print(x)

S2 = Surjection(2)
u = S2([1, 2, 1])
print(u)
print(u.section())
print(u.section().table_reduction())
