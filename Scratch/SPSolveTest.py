import jax
import jax.numpy as np

def A(x):
   y = 2*x
   ind = np.array([1, 2])
   y = y.at[ind].set(x[ind], unique_indices=True)
   return y

b = np.ones(10)
# Yet, jax.scipy.sparse.linalg.cg will NOT throw an exception
sol, info = jax.scipy.sparse.linalg.bicgstab(A, b)