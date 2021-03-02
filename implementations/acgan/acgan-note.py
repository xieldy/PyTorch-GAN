import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("MNIST", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m): # 权重初始化函数
    classname = m.__class__.__name__ # 得到当前Module类的名称
    if classname.find("Conv") != -1: # 类名称中包含"Conv"
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) # 正态分布，均值=0，标准差=0.02
    elif classname.find("BatchNorm2d") != -1: # 类名称中包含"BatchNorm2d"
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02) 
        torch.nn.init.constant_(m.bias.data, 0.0) # 初始化为常数


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__() # 继承父类__init__方法

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim) # 将标签值映射到对应的embedding向量

        self.init_size = opt.img_size // 4  # Initial size before upsampling，确保图像在经过conv_blocks的两次上采样之后刚好等于原始图像大小(img_size)
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2)) #第一个线性层，输出size等于128个下采样图像（经过两次上采样后即为原始大小图像）

        self.conv_blocks = nn.Sequential( # 卷积模块
            nn.BatchNorm2d(128), # 归一化函数，改变数据的量纲
            nn.Upsample(scale_factor=2), # 上采样函数，放大两倍
            nn.Conv2d(128, 128, 3, stride=1, padding=1), # 二维卷积函数，输入参数为：input_channel=128, output_channel=128, kennel_size=3, stride(卷积步长)=3, padding(零填充)=1
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True), # 带有倾斜角度的激活函数
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise) # 将噪音和标签混合
        out = self.l1(gen_input) # 输入L1层
        out = out.view(out.shape[0], 128, self.init_size, self.init_size) # 对于L1层的输出进行维度变换，out.shape[0]等于batch_size
        img = self.conv_blocks(out) # 将维度变换后的输出输入conv_blocks
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img) # 将输入的样本传递到conv_blocks
        out = out.view(out.shape[0], -1) # 输出的值进行维度变换，准备输入线性层
        validity = self.adv_layer(out) # 输入线性层，输出 真/假 判断
        label = self.aux_layer(out) # 输出预测向量，输出 类别 判断

        return validity, label #输出真/假和类别标签


# Loss functions
adversarial_loss = torch.nn.BCELoss() # 定义对抗真假损失函数
auxiliary_loss = torch.nn.CrossEntropyLoss() # 定义辅助分类损失函数

# Initialize generator and discriminator
generator = Generator() # 实例化一个生成器对象
discriminator = Discriminator() # 实例化一个判别器对象

if cuda: # 损失函数和模型对象都装载到CUDA
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal) # apply函数会递归地搜索网络内的所有Module并把参数表示的函数应用到所有的Module上
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs("/home/data/mnist", exist_ok=True) # 建立数据集目录
dataloader = torch.utils.data.DataLoader( # 生成数据加载器
    datasets.MNIST(
        "/home/data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])] # transforms.Resize 图像按比例缩放到指定尺寸
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    pin_memory=True, # pin_memory 锁页内存，在显存足够大的时候能提升运算效率
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)) # 定义优化器
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor # 根据CUDA可用状态定义tensor变量类型
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs): # 开始训练，总轮数等于opt.n_epochs
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0] # 定义batch_size

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False) # 定义ground truth，后面用于计算损失
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor)) # 将dataloader输出的图像和标签转变为CUDA或无CUDA的变量
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad() # 将优化器的梯度历史置零

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim)))) # 随机生成噪音
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size))) # 随机生成目标标签

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels) # 生成器生成样本

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_imgs) #判别器判别生成的样本
        g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels)) #计算生成器损失

        g_loss.backward() # 反向传播
        optimizer_G.step() # 参数更新

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad() # 梯度置零

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs) # 判别真样本
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2 # 计算真样本损失

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach()) # 判别假样本，detach()用于截断反向传播，防止在对d_loss反向传播的时候影响生成器的参数
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2 # 计算假样本损失

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2 # 计算总损失

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt) # 计算判别器精确度

        d_loss.backward() # 反向传播
        optimizer_D.step() # 梯度更新

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
        )
        batches_done = epoch * len(dataloader) + i 
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
