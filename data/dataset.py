import torch

a = [[1], [2], [3], [4], [5]]
x = torch.tensor(a, dtype=torch.float32)

y = 2 * x

assert x.shape == y.shape
assert x.dtype == torch.float32
assert y.dtype == torch.float32

assert torch.equal(x * 2, y)
