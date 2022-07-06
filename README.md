# Keras-DDPM
生成扩散模型的Keras实现

## 介绍
- 博客：https://kexue.fm/archives/9152
- 博客：https://kexue.fm/archives/9119

## 说明
- 模型主体依然式U-Net格式，但是经过自己的简化（如特征拼接改为相加、去掉了Attention等），加快了收敛速度；
- 在单张3090下，训练半天可以初见效果，训练3天的效果如下
<img src="https://kexue.fm/usr/uploads/2022/07/3342802728.png" width=560>

## 环境
- tensorflow 1.15
- keras 2.3.1
- bert4keras 0.11.3

## 要点

## 交流
QQ交流群：808623966，微信群请加机器人微信号spaces_ac_cn
