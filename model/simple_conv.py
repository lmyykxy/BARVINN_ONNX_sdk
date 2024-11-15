import numpy as np
import torch
import torch.nn as nn

class SimpleConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, groups, dilation, wprec):
        super(SimpleConv, self).__init__()

        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)

        # import ipdb as pdb; pdb.set_trace()
        max_int = (2**wprec) - 1 # 2位bit的最大表示数为3
        w_data = np.random.randint(max_int+1, size=(in_ch*out_ch*kernel_size*kernel_size)) # 生成随机数组,[0,max_int]
        weights = np.asarray(w_data).astype(np.float32).reshape(out_ch, in_ch,kernel_size,kernel_size) # 数组格式调整为64 x 64的3x3
        self.conv1.weight.data = torch.from_numpy(weights)

    def forward(self, x):
        out = self.conv1(x)
        return out

def export_torch_to_onnx(model, batch_size, nb_channels, w, h):
    if isinstance(model, torch.nn.Module):
        model_name =  model.__class__.__name__
        # create the imput placeholder for the model
        # note: we have to specify the size of a batch of input images
        input_placeholder = torch.randn(batch_size, nb_channels, w, h)
        onnx_model_fname = model_name + ".onnx"
        # export pytorch model to onnx
        print(model)
        torch.onnx.export(model, input_placeholder, onnx_model_fname)
        print("{0} was exported to onnx: {1}".format(model_name, onnx_model_fname))
        return onnx_model_fname
    else:
        print("Unsupported model file")
        return

'''
in_ch			: 输入通道数
out_ch			: 输出通道数
kernel_size		: 卷积核的大小 (单个整数,如3表示3X3的卷积核)
stride			: 步幅,即卷积核在输入上滑动的步长
padding			: 填充,决定在输入的每一边补充的像素数
dilation		: 扩张率,控制卷积核元素之间的距离。默认为 1(不扩张).增大扩张率会在卷积核元素间引入"空隙",从而增加感受野
groups			: 控制输入和输出通道的连接方式.默认值为 1,表示所有输入通道都与所有输出通道相连.
				  设置为 in_channels 值时，可以实现“深度卷积”（每个输入通道独立应用一个卷积核）
bias			: 是否使用偏置项.默认为 True,表示为卷积的每个输出通道添加一个可学习的偏置
'''
if __name__ == '__main__':
    input_size = 32
    in_ch = 64
    out_ch= 64
    kernel_size = 3
    stride = 1
    padding = 1
    groups = 1
    dilation = 1
    wprec = 2
    model = SimpleConv(in_ch, out_ch, kernel_size, stride, padding, groups ,dilation, wprec)
    # import ipdb as pdb; pdb.set_trace()
    input_tensor = torch.randint(0,255, [1, in_ch, input_size,input_size]).type(torch.float32) 
    # print(model(input_tensor))
    export_torch_to_onnx(model, 1, in_ch, input_size,input_size)
