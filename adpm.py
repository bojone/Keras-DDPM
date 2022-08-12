#! -*- coding: utf-8 -*-
# 生成扩散模型Analytic-DPM参考代码
# 在DDIM上修改，不用改变训练，只修改采样过程的方差
# 博客：https://kexue.fm/archives/9245

# from ddpm import *  # 加载训练好的模型
from ddpm2 import *  # 加载训练好的模型


def data_generator(t=0):
    """图片读取
    """
    batch_imgs = []
    while True:
        for i in np.random.permutation(len(imgs)):
            batch_imgs.append(imread(imgs[i]))
            if len(batch_imgs) == batch_size:
                batch_imgs = np.array(batch_imgs)
                batch_steps = np.array([t] * batch_size)
                batch_bar_alpha = bar_alpha[batch_steps][:, None, None, None]
                batch_bar_beta = bar_beta[batch_steps][:, None, None, None]
                batch_noise = np.random.randn(*batch_imgs.shape)
                batch_noisy_imgs = batch_imgs * batch_bar_alpha + batch_noise * batch_bar_beta
                yield [batch_noisy_imgs, batch_steps[:, None]]
                batch_imgs = []


factors = [(model.predict(data_generator(t), steps=5)**2).mean()
           for t in tqdm(range(T), ncols=0)]  # 用(batch_size * steps)个样本去估计方差修正项
factors = np.clip(1 - np.array(factors), 0, 1)


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
    gamma_ = epsilon_ * bar_alpha_pre_ / bar_alpha_  # 增加代码
    sigma_ = np.sqrt(sigma_**2 + gamma_**2 * factors[::stride])  # 增加代码
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


sample('test.png', n=8, stride=100, eta=1)
