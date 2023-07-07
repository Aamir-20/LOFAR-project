# -*- coding: utf-8 -*-
import torch

# ================================================================= #
#                        Initializing Tensor                         #
# ================================================================= #


device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device, requires_grad=True)

# print(my_tensor)
# print(my_tensor.dtype)
# print(my_tensor.device)
# print(my_tensor.shape)
# print(my_tensor.requires_grad)

# Other common initialization methods.
x = torch.empty(size=(3, 3))
# print(x)
x = torch.zeros((3, 3))
# print(x)
x = torch.rand((3, 3))
# print(x)
x = torch.ones((3, 3))
# print(x)
x = torch.eye(5, 5) # I, eye (a way of remembering identity matrix)
# print(x)
x = torch.arange(start=0, end=5, step=1)
# print(x)
x = torch.linspace(start=0.1, end=1, steps=10)
# print(x)
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
# print(x)
x = torch.empty(size=(1, 5)).uniform_(0, 1)
# print(x)
x = torch.diag(torch.ones(3))
# print(x)

# How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)
# print(tensor)
# print(tensor.bool()) # `0` is True, otherwise False
# print(tensor.short()) # changes values to int16
# print(tensor.long()) # changes values to int64 
# print(tensor.half()) # changes to float16
# print(tensor.float()) # changes to float32
# print(tensor.double()) # changes to float64

# Array to Tensor conversion and vice-versa.
import numpy as np
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()


# ================================================================= #
#               Tensor Math & Comparison Operations                 #
# ================================================================= #

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition.
z1 = torch.empty(3)
torch.add(x, y, out=z1)
# print(z1)

z2 = torch.add(x, y)
# print(z2)

z = x + y
# print(z) 

# Subtraction.
z = x - y
# print(z)

# Divsion.
z = torch.true_divide(x, y) # element-wise division if same shape
# print(z)

# Inplace Operations.
t = torch.zeros(3)
t.add_(x) # equivalent to `t += x` but not equivalent to `t = t + x`
# print(t)

# Exponentiation.
z = x.pow(2) # raise each element to the power of 2
# print(z)
z = x ** 2 # equivalent to `x.pow(2)`

# Simple Comparison.
z = x > 0
# print(z)
z = x < 0
# print(z)

# Matrix Multiplication.
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2) # resulting in a 2x3 matrix
x3 = x1.mm(x2) # equivalent to above

# Matrix Exponentiation.
matrix_exp = torch.rand(5, 5)
# print(matrix_exp.matrix_power(3))

# Element-wise Multiplication.
z = x * y
# print(z)

# Dot Product.
z = torch.dot(x, y)
# print(z)

# Batch Matrix Multiplication.
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
# print(tensor1)
tensor2 = torch.rand((batch, m, p))
# print(tensor2)
out_bmm = torch.bmm(tensor1, tensor2) # (batch, n, p)
# print(out_bmm)

# Example of Broadcasting.
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))
# print(x1)
# print(x2)
z = x1 - x2 # it will match the rows and columns to complete operation
# print(z)
z = x1 ** x2
# print(z)

# Other useful tensor operations.
sum_x = torch.sum(x, dim=0) # x.sum(dim=0)
values, indices = torch.max(x, dim=0) # x.max(dim=0)
values, indices = torch.min(x, dim=0) # x.min(dim=0)
abs_x = torch.abs(x) # x.abs(dim=0)
z = torch.argmax(x, dim=0) # x.argmax(dim=0)
z = torch.argmin(x, dim=0) # x.argmin(dim=0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x, y) # element-wise comparison
# print(z)    
sorted_y, indices = torch.sort(y, dim=0, descending=False)
# print(sorted_y, indices)

z = torch.clamp(x, min=0, max=10) # if any value is below 0 it will be 
#set to 0 and if any value is above ten it will be set to 10.

x = torch.tensor([1,0,1,1,1], dtype=torch.bool)
z = torch.any(x)
# print(z)
z = torch.all(x) # All values must be `1`.
# print(z)
 


# ================================================================= #
#                        Tensor Indexing                            #
# ================================================================= #

batch_size = 10
features = 25
x = torch.rand((batch_size, features))
# print(x) # a 10x25 matrix, 10 rows with 25 elements in each row.
# print(x.shape)

# print(x[0].shape) # equivalent to x[0,:]

# print(x[:,0].shape)

# print(x[2, 0:10]) # 0:10 --> [0, 1, 2, ..., 9], third row of all the elements up to 10.

x[0, 0] = 100

# Fancy Indxing
x = torch.arange(10)
indices = [2, 5, 8]
# print(x[indices])

x = torch.rand((3, 5))
# print(x)
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
# print(x[rows, cols]) # pick out the (2,5) and (1,1) entry from x. 
# print(x[rows, cols].shape)

# More advanced indexing.
x = torch.arange(10)
# print(x)
# print(x[(x < 2) | (x > 8)]) # can replace or `|` with an and sign `&`.
# print(x[x.remainder(2) == 0]) # pick out all the even elements.

# Useful operations.
# print(torch.where(x > 5, x, x*2)) # if x>5 print x otherwise print 2*x.
# print(torch.tensor([0,0,1,2,2,3,4]).unique()) # removes repeats.
# print(x.ndimension()) # A 5x5x5 would result in 3 as the output.
# print(x.numel()) # counts the number of elements in `x`.



# ================================================================= #
#                          Tensor Reshaping                         #
# ================================================================= #

x = torch.arange(9)

x_3x3 = x.view(3, 3)
# print(x_3x3)
x_3x3 = x.reshape(3, 3) # same as `view` but stored differently.
# print(x_3x3)            # `reshape` is the safe bet, but performance loss.

y = x_3x3.t()
# print(y)    
# print(y.contiguous().view(9))

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
# print(torch.cat((x1, x2), dim=0).shape)
# print(torch.cat((x1, x2), dim=1).shape)

z = x1.view(-1)
# print(z.shape)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
# print(z.shape)

z = x.permute(0, 2, 1)
# print(z.shape)

x = torch.arange(10)
# print(x.unsqueeze(0).shape)
# print(x.unsqueeze(1).shape)

x = torch.arange(10).unsqueeze(0).unsqueeze(1) # 1x1x10
# print(x)

z = x.squeeze(1)
# print(z)
# print(z.shape)






