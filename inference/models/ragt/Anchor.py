import numpy as np


anchor_thetas = [x*0.2094 for x in range(15)]
# anchor宽
anchor_w = 85.72
# anchor高
anchor_h = 19.15
# 每个grid cell 的anchor数
num_anchors = 3
# 输出层下采样次数
times_of_down_sampling = 5
# 输入图像尺寸
img_size = 416
# 防止角度偏移量为0
Anchor_eps = 0.000001


field_of_grid_cell = 2 ** times_of_down_sampling
num_grid_cell = int(img_size / field_of_grid_cell)
theta_margin = 180 / num_anchors


# if __name__ == '__main__':
#     print(field_of_grid_cell)
#     print(anchor_thetas)
#     print(3//0.2094)
#     a = np.arange(16).reshape((4, 4))
#     print(33.2%16.4)
#     a = np.arange(10*26*26*15*6).reshape((10, 26, 26, 15, 6))
#     b = []
#     for i in a:
#         b.append(i)
#     # b = np.array(b)
#     print(type(b[0]))
#     # print(b.shape)
#     c = np.arange(16).reshape((4, 4))
#     d = []
#     for i in c:
#         d.append(i)
#     print(d)
#     print(np.array(d))
