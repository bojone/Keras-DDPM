#! -*- coding: utf-8 -*-
# DDCM（Denoising Diffusion Codebook Models）参考代码
# 在DDPM上修改，不用改变训练，只修改采样过程
# 博客：https://kexue.fm/archives/9245

from ddpm2 import *  # 加噪训练好的模型

K = 64  # 每步的Codebook大小
codebook = np.random.randn(T + 1, K, img_size, img_size, 3)


def sample(path, n=4):
    """随机采样函数
    """
    z_samples = codebook[T][np.random.choice(K, size=n**2)]
    for t in tqdm(range(T), ncols=0):
        t = T - t - 1
        bt = np.array([[t]] * z_samples.shape[0])
        z_samples -= beta[t]**2 / bar_beta[t] * model.predict([z_samples, bt])
        z_samples /= alpha[t]
        z_samples += codebook[t][np.random.choice(K, size=n**2)] * sigma[t]
    x_samples = np.clip(z_samples, -1, 1)
    figure = np.zeros((img_size * n, img_size * n, 3))
    for i in range(n):
        for j in range(n):
            digit = x_samples[i * n + j]
            figure[i * img_size:(i + 1) * img_size,
                   j * img_size:(j + 1) * img_size] = digit
    imwrite(path, figure)


def encode(path, n=4):
    """随机选一些图片，进行编码和重构
    """
    x_samples = [imread(f) for f in np.random.choice(imgs, n**2)]
    z_samples = np.repeat(codebook[T][:1], n**2, axis=0)
    for t in tqdm(range(T), ncols=0):
        t = T - t - 1
        bt = np.array([[t]] * z_samples.shape[0])
        mp = model.predict([z_samples, bt])
        x0 = (z_samples - bar_beta[t] * mp) / bar_alpha[t]
        sims = np.einsum('kuwv,buwv->kb', codebook[t], x_samples - x0)
        idxs = sims.argmax(0)
        z_samples -= beta[t]**2 / bar_beta[t] * mp
        z_samples /= alpha[t]
        z_samples += codebook[t][idxs] * sigma[t]
    z_samples = np.clip(z_samples, -1, 1)
    figure = np.zeros((img_size * n, img_size * n * 2, 3))
    for i in range(n):
        for j in range(n):
            digit = x_samples[i * n + j]
            figure[i * img_size:(i + 1) * img_size,
                   2 * j * img_size:(2 * j + 1) * img_size] = digit
            digit = z_samples[i * n + j]
            figure[i * img_size:(i + 1) * img_size,
                   (2 * j + 1) * img_size:(2 * j + 2) * img_size] = digit
    imwrite(path, figure)


sample(f'test1.png')
encode(f'test2.png')
