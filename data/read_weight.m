% 读取 .mat 文件
WeightdataStruct = load('weight_file.mat');
InputdataStruct = load('input_file.mat');

% 提取数据
Weightdata = WeightdataStruct.data;
Inputdata = squeeze(InputdataStruct.data);
MData = Inputdata * Weightdata;
print('done')