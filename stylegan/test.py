# import torch

# a = torch.randn((1, 512))
# b = torch.randn((1, 512))
# c = torch.cat([a, b], dim= 1)
# print(c.size())

def f(a):
    for i in range(len(a)):
        a[i] = a[i] + 1
    return a

a = [1, 2, 3]
print(a)
b = f(a)
print(a)
print(b)