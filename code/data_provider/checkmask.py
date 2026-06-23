import numpy as np


feature="v10"
filename_mean = f"../scaler/{feature}_mean.npy"
filename_std = f"../scaler/{feature}_std.npy"
filename_mean_e = f"../scaler/{feature}_mean_e.npy"
filename_std_e = f"../scaler/{feature}_std_e.npy"
a=np.load(filename_mean)
b=np.load(filename_std)
c=np.load(filename_mean_e)
d=np.load(filename_std_e)
print(a,b,c,d)