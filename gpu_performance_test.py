import torch
import timeit

# cpu
device = torch.device("cpu")
x = torch.rand((10000,10000), dtype=torch.float32)
y = torch.rand((10000,10000), dtype=torch.float32)
x = x.to(device)
y = y.to(device)

#%%timeit
def test():
    return x * y
t1 = timeit.timeit('test()', setup='from __main__ import test', number=100)
print(t1)


# gpu
device = torch.device("mps")
x = torch.rand((10000,10000), dtype=torch.float32)
y = torch.rand((10000,10000), dtype=torch.float32)
x = x.to(device)
y = y.to(device)

t2 = timeit.timeit('test()', setup='from __main__ import test', number=100)
print(t2)
