#! -*- coding: utf-8 -*-
# 生成扩散模型DDIM参考代码
# DDIM不用改变训练，只修改采样过程
# 博客：https://kexue.fm/archives/9181

# from ddpm import *  # 加载训练好的模型
from ddpm2 import *  # 加载训练好的模型


def sample(path=None, n=4, z_samples=None, stride=1, eta=1):
    """随机采样函数
    注：eta控制方差的相对大小；stride空间跳跃
    """
    # 采样参数
    bar_alpha_ = bar_alpha[::stride]
    bar_alpha_pre_ = np.pad(bar_alpha_[:-1], [1, 0], constant_values=1)
    bar_beta_ = np.sqrt(1 - bar_alpha_**2)
    bar_beta_pre_ = np.sqrt(1 - bar_alpha_pre_**2)
    alpha_ = bar_alpha_ / bar_alpha_pre_
    sigma_ = bar_beta_pre_ / bar_beta_ * np.sqrt(1 - alpha_**2) * eta
    epsilon_ = bar_beta_ - alpha_ * np.sqrt(bar_beta_pre_**2 - sigma_**2)
    T_ = len(bar_alpha_)
    # 采样过程
    if z_samples is None:
        z_samples = np.random.randn(n**2, img_size, img_size, 3)
    else:
        z_samples = z_samples.copy()
    for t in tqdm(range(T_), ncols=0):
        t = T_ - t - 1
        bt = np.array([[t * stride]] * z_samples.shape[0])
        z_samples -= epsilon_[t] * model.predict([z_samples, bt])
        z_samples /= alpha_[t]
        z_samples += np.random.randn(*z_samples.shape) * sigma_[t]
    x_samples = np.clip(z_samples, -1, 1)
    if path is None:
        return x_samples
    figure = np.zeros((img_size * n, img_size * n, 3))
    for i in range(n):
        for j in range(n):
            digit = x_samples[i * n + j]
            figure[i * img_size:(i + 1) * img_size,
                   j * img_size:(j + 1) * img_size] = digit
    imwrite(path, figure)


def sample_inter(path, n=4, k=8, stride=1):
    """随机采样插值函数
    说明：随机选择两个随机向量进行球面均匀插值，然后生成对应的结果。
    """
    figure = np.ones((img_size * n, img_size * k, 3))
    Z = np.random.randn(n * 2, img_size, img_size, 3)
    z_samples = []
    for i in range(n):
        for j in range(k):
            theta = np.pi / 2 * j / (k - 1)
            z = Z[2 * i] * np.sin(theta) + Z[2 * i + 1] * np.cos(theta)
            z_samples.append(z)
    x_samples = sample(z_samples=np.array(z_samples), stride=stride, eta=0)
    for i in range(n):
        for j in range(k):
            ij = i * k + j
            figure[i * img_size:(i + 1) * img_size,
                   img_size * j:img_size * (j + 1)] = x_samples[ij]
    imwrite(path, figure)


sample('test.png', n=4, stride=100, eta=0)
sample_inter('test_inter.png', n=8, k=15, stride=20)
