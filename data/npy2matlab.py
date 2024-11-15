import numpy as np
from scipy.io import savemat

# 读取 .npy 文件
# npy_data = np.load('./data/onnx__MatMul_4.npy')
npy_data = np.load('./data/input_tensor.npy')

# 将数据保存为 .mat 文件
savemat('input_file.mat', {'data': npy_data})
