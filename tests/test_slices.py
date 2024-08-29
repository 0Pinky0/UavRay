import torch

a = torch.zeros([32, 4146])

r_shape = (16, 16, 16)

b = a[:, :50]
print(b.shape)
c = a[:, 50:]
print(c.shape)
c = c.reshape(32, *r_shape)
print(c.shape)
