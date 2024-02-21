#! -*- coding: utf-8 -*-
# 生成扩散模型DDPM参考代码
# 用了Pre Norm GAU架构
# 实验环境：tf 1.15 + keras 2.3.1 + bert4keras
# 参考：https://kexue.fm/archives/9984

import os
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import *
from keras.callbacks import Callback
from keras_preprocessing.image import list_pictures
from bert4keras.layers import *
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_layer_adaptation
from bert4keras.optimizers import extend_with_piecewise_linear_lr
from bert4keras.optimizers import extend_with_exponential_moving_average
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")  # 忽略keras带来的满屏警告

if not os.path.exists('samples'):
    os.mkdir('samples')

# 基本配置
imgs = list_pictures('/mnt/vepfs/sujianlin/CelebA-HQ/train/', 'png')
imgs += list_pictures('/mnt/vepfs/sujianlin/CelebA-HQ/valid/', 'png')
np.random.shuffle(imgs)
img_size = 128  # 如果只想快速实验，可以改为64
batch_size = 64  # 如果显存不够，可以降低为32、16，但不建议低于16
hidden_size = 768
num_layers = 24

# 超参数选择
T = 1000
alpha = np.sqrt(1 - 0.02 * np.arange(1, T + 1) / T)
beta = np.sqrt(1 - alpha**2)
bar_alpha = np.cumprod(alpha)
bar_beta = np.sqrt(1 - bar_alpha**2)
sigma = beta.copy()
# sigma *= np.pad(bar_beta[:-1], [1, 0]) / bar_beta


def imread(f, crop_size=None):
    """读取图片
    """
    x = cv2.imread(f)
    height, width = x.shape[:2]
    if crop_size is None:
        crop_size = min([height, width])
    else:
        crop_size = min([crop_size, height, width])
    height_x = (height - crop_size + 1) // 2
    width_x = (width - crop_size + 1) // 2
    x = x[height_x:height_x + crop_size, width_x:width_x + crop_size]
    if x.shape[:2] != (img_size, img_size):
        x = cv2.resize(x, (img_size, img_size))
    x = x.astype('float32')
    x = x / 255 * 2 - 1
    return x


def imwrite(path, figure):
    """归一化到了[-1, 1]的图片矩阵保存为图片
    """
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    cv2.imwrite(path, figure)


def data_generator():
    """图片读取
    """
    batch_imgs = []
    while True:
        for i in np.random.permutation(len(imgs)):
            batch_imgs.append(imread(imgs[i]))
            if len(batch_imgs) == batch_size:
                batch_imgs = np.array(batch_imgs)
                batch_steps = np.random.choice(T, batch_size)
                batch_bar_alpha = bar_alpha[batch_steps][:, None, None, None]
                batch_bar_beta = bar_beta[batch_steps][:, None, None, None]
                batch_noise = np.random.randn(*batch_imgs.shape)
                batch_noisy_imgs = batch_imgs * batch_bar_alpha + batch_noise * batch_bar_beta
                yield [batch_noisy_imgs, batch_steps[:, None]], batch_noise
                batch_imgs = []


def rope_2d(x):
    """2D-RoPE
    """
    w = img_size // 8
    pos = K.arange(0, w**2, dtype='float32')
    pos1, pos2 = pos // w, pos % w
    pos1 = sinusoidal_embeddings(pos1, 64, 1000)
    pos2 = sinusoidal_embeddings(pos2, 64, 1000)
    return K.concatenate([pos1, pos2], 1)[None]


def l2_loss(y_true, y_pred):
    """用l2距离为损失，不能用mse代替
    """
    return K.sum((y_true - y_pred)**2, axis=[1, 2, 3])


# 搭建去噪模型
x_in = x = Input(shape=(img_size, img_size, 3))
x = Reshape((img_size // 8, 8, img_size // 8, 8, 3))(x)
x = Permute((1, 3, 2, 4, 5))(x)
x = Reshape((img_size**2 // 64, 192))(x)
x = Dense(hidden_size, use_bias=False)(x)

t_in = Input(shape=(1,))
t = Embedding(input_dim=T, output_dim=hidden_size)(t_in)

x = Add()([x, t])
p = Lambda(rope_2d)(x)

for i in range(num_layers):
    xi = x
    x = LayerNormalization(zero_mean=False, offset=False)(x)
    x = GatedAttentionUnit(hidden_size * 2, 128, normalization='softmax')([x, p], p_bias='rotary')
    x = Add()([xi, x])

x = LayerNormalization(zero_mean=False, offset=False)(x)
x = Dense(192, use_bias=False)(x)
x = Reshape((img_size // 8, img_size // 8, 8, 8, 3))(x)
x = Permute((1, 3, 2, 4, 5))(x)
x = Reshape((img_size, img_size, 3))(x)

model = Model(inputs=[x_in, t_in], outputs=x)
model.summary()

OPT = extend_with_layer_adaptation(Adam)
OPT = extend_with_piecewise_linear_lr(OPT)  # 此时就是LAMB优化器
OPT = extend_with_exponential_moving_average(OPT)  # 加上滑动平均
optimizer = OPT(
    learning_rate=1e-3,
    ema_momentum=0.9999,
    exclude_from_layer_adaptation=['Norm', 'bias'],
    lr_schedule={
        4000: 1,  # Warmup步数
        20000: 0.5,
        40000: 0.1,
    }
)
model.compile(loss=l2_loss, optimizer=optimizer)


def sample(path=None, n=4, z_samples=None, t0=0):
    """随机采样函数
    """
    if z_samples is None:
        z_samples = np.random.randn(n**2, img_size, img_size, 3)
    else:
        z_samples = z_samples.copy()
    for t in tqdm(range(t0, T), ncols=0):
        t = T - t - 1
        bt = np.array([[t]] * z_samples.shape[0])
        z_samples -= beta[t]**2 / bar_beta[t] * model.predict([z_samples, bt])
        z_samples /= alpha[t]
        z_samples += np.random.randn(*z_samples.shape) * sigma[t]
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


def sample_inter(path, n=4, k=8, sep=10, t0=500):
    """随机采样插值函数
    """
    figure = np.ones((img_size * n, img_size * (k + 2) + sep * 2, 3))
    x_samples = [imread(f) for f in np.random.choice(imgs, n * 2)]
    X = []
    for i in range(n):
        figure[i * img_size:(i + 1) * img_size, :img_size] = x_samples[2 * i]
        figure[i * img_size:(i + 1) * img_size,
               -img_size:] = x_samples[2 * i + 1]
        for j in range(k):
            lamb = 1. * j / (k - 1)
            x = x_samples[2 * i] * (1 - lamb) + x_samples[2 * i + 1] * lamb
            X.append(x)
    x_samples = np.array(X) * bar_alpha[t0]
    x_samples += np.random.randn(*x_samples.shape) * bar_beta[t0]
    x_rec_samples = sample(z_samples=x_samples, t0=t0)
    for i in range(n):
        for j in range(k):
            ij = i * k + j
            figure[i * img_size:(i + 1) * img_size, img_size * (j + 1) +
                   sep:img_size * (j + 2) + sep] = x_rec_samples[ij]
    imwrite(path, figure)


class Trainer(Callback):
    """训练回调器
    """
    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('model.weights')
        sample('samples/%05d.png' % (epoch + 1))
        optimizer.apply_ema_weights()
        model.save_weights('model.ema.weights')
        sample('samples/%05d_ema.png' % (epoch + 1))
        optimizer.reset_old_weights()


if __name__ == '__main__':

    trainer = Trainer()
    model.fit(
        data_generator(),
        steps_per_epoch=2000,
        epochs=10000,  # 只是预先设置足够多的epoch数，可以自行Ctrl+C中断
        callbacks=[trainer]
    )

else:

    model.load_weights('model.ema.weights')
