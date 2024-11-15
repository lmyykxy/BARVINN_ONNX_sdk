# 读取十六进制文件并逐行转换为二进制
with open('output.hex', 'r') as file:
    lines = file.readlines()

# 将每行的十六进制字符串转换为二进制字符串
binary_lines = [bin(int(line.strip(), 16))[2:].zfill(len(line.strip()) * 4) for line in lines]

# 将转换后的二进制字符串写入到新文件
with open('output_binary.txt', 'w') as output_file:
    for binary in binary_lines:
        output_file.write(binary + '\n')
