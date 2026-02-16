from uconf import *

S1 = Surjection(1)
S2 = Surjection(2)
S1_1 = tensor([S1, S1])


def ff(x):
    a, b = x
    print(a)
    print(type(a))
    print(b)
    print(type(x))
    return S2((1, 1))


f = S1_1.module_morphism(on_basis=ff, codomain=S2)
x = S1([1])
z = f(tensor([x, x]))
print(z)
print(z.parent())
print(type(z))
print(z + x)
