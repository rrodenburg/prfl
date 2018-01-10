import numpy as np

mat1 = np.zeros((32,3))
mat2 = np.zeros((3,32))

prod1 = np.dot(mat1,mat2)
prod2 = np.dot(mat2,mat1)

print(prod1.shape)
print(prod2.shape)


