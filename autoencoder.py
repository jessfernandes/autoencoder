import os
import time
from contextlib import suppress

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms


def save_img(img, name):
    global dimensions
    global means

    npimg = img.numpy()
    # npimg[0] = npimg[0] * stds[0] + means[0]
    # npimg[1] = npimg[1] * stds[1] + means[1]
    # npimg[2] = npimg[2] * stds[2] + means[2]

    npimg = np.transpose(npimg, (1, 2, 0))

    plt.imsave(name, npimg)


class PrintLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.size())
        return x


class GaussianNoiser():
    def __init__(self, noise_amount):
        self.noise_amount = noise_amount

    def __call__(self, imgs):
        noisy_imgs = imgs.clone()

        # gaussian noise
        noise = torch.rand(noisy_imgs.size())

        noisy_imgs += self.noise_amount * noise

        # clamp between 0 and 1
        noisy_imgs.clamp_(0, 1)

        return noisy_imgs


class DropPixelNoiser():
    def __init__(self, noise_chance):
        self.noise_chance = noise_chance

    def __call__(self, imgs):
        noisy_imgs = imgs.clone()
        # noise described in the denoising autoencoder paper from bengio
        for img in noisy_imgs:
            for i in range(img.size(1)):
                for j in range(img.size(2)):
                    if np.random.random() < self.noise_chance:
                        img[:, i, j] = 0
        return noisy_imgs


class EncoderBlock(nn.Module):
    def __init__(self, input_size, output_size,
                 last_activation, apply_bn_last=True):
        super().__init__()

        self.apply_bn_last = apply_bn_last

        self.conv1 = nn.Conv2d(input_size, output_size,
                               kernel_size=5,
                               stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(output_size)

        self.conv2 = nn.Conv2d(output_size, output_size,
                               kernel_size=5,
                               stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(output_size)

        self.conv3 = nn.Conv2d(output_size, output_size,
                               kernel_size=3,
                               stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(output_size)

        self.conv4 = nn.Conv2d(output_size, output_size,
                               kernel_size=3,
                               stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(output_size)

        self.conv5 = nn.Conv2d(output_size, output_size,
                               kernel_size=3,
                               stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(output_size)

        self.downsampler = nn.Conv2d(output_size, output_size,
                                     kernel_size=2,
                                     stride=2, padding=0)

        if apply_bn_last:
            self.bn_down = nn.BatchNorm2d(output_size)

        self.last_activation = last_activation()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = F.relu(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = F.relu(out)

        out = self.downsampler(out)
        if self.apply_bn_last:
            out = self.bn_down(out)
        out = self.last_activation(out)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, input_size, output_size,
                 last_activation, apply_bn_last=True):
        super().__init__()

        self.apply_bn_last = apply_bn_last
        self.last_activation = last_activation

        self.conv1 = nn.Conv2d(input_size, output_size,
                               kernel_size=5,
                               stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(output_size)

        self.conv2 = nn.Conv2d(output_size, output_size,
                               kernel_size=5,
                               stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(output_size)

        self.conv3 = nn.Conv2d(output_size, output_size,
                               kernel_size=3,
                               stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(output_size)

        self.conv4 = nn.Conv2d(output_size, output_size,
                               kernel_size=3,
                               stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(output_size)

        self.conv5 = nn.Conv2d(output_size, output_size,
                               kernel_size=3,
                               stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(output_size)

        self.upsampler = nn.ConvTranspose2d(output_size, output_size,
                                            kernel_size=2,
                                            stride=2, padding=0)
        if apply_bn_last:
            self.bn_up = nn.BatchNorm2d(output_size)

        if last_activation is not None:
            self.last_activation = last_activation()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = F.relu(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = F.relu(out)

        out = self.upsampler(out)
        if self.apply_bn_last:
            out = self.bn_up(out)

        if self.last_activation is not None:
            out = self.last_activation(out)

        return out


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            EncoderBlock(3, 16, last_activation=nn.ReLU),

            EncoderBlock(16, 8, last_activation=nn.ReLU),

            EncoderBlock(8, 4, last_activation=nn.ReLU)
        )

        # 7 x 7 x 7 representation

        self.decoder = nn.Sequential(
            DecoderBlock(4, 8, last_activation=nn.ReLU),

            DecoderBlock(8, 16, last_activation=nn.ReLU),

            DecoderBlock(16, dimensions, last_activation=nn.Sigmoid,
                         apply_bn_last=False)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(device, model, train_data, test_data, epochs, optimiser,
          criterion, model_name='model.bin'):
    # try to load model
    try:
        checkpoint = torch.load(model_name)
        initial_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        best_loss = checkpoint['best_loss']
        best_model_state_dict = checkpoint['best_model_state_dict']
    # file does not exist or pytorch error (model architecture changed)
    except:
        initial_epoch = 0
        best_loss = None

    noiser = GaussianNoiser(0.10)
    for epoch in range(initial_epoch, epochs):
        print('Started epoch {}'.format(epoch))
        start_time = time.perf_counter()
        if epoch != 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimiser_state_dict': optimiser.state_dict(),
                        'best_loss': best_loss,
                        'best_model_state_dict': best_model_state_dict,
                        }, model_name)

        save = True
        train_loss = 0
        with torch.set_grad_enabled(True):
            # set model for training
            model.train()

            for imgs, _ in train_data:
                imgs = imgs.to(device)

                noisy_imgs = noiser(imgs)

                output = model(noisy_imgs)

                loss = criterion(output, imgs)
                print(loss)

                # zero gradients per batch
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # name, p = list(model.named_parameters())[-1]
                # print(name, p.grad)

                # compute total loss
                train_loss += loss.item() * imgs.size(0)

                if epoch % 5 == 0 and save:
                    # sample two images
                    for j in range(2):
                        save_img(imgs[j].detach(), 'result_imgs_train/epoch-{}-{}-original.jpg'.format(epoch, j))
                        save_img(output[j].detach(), 'result_imgs_train/epoch-{}-{}-mod.jpg'.format(epoch, j))
                    save = False

        save = True
        val_loss = 0
        with torch.set_grad_enabled(False):
            # set model for evaluation
            model.eval()

            for imgs, _ in val_data:
                imgs = imgs.to(device)

                output = model(imgs)

                loss = criterion(output, imgs)

                # compute total loss
                val_loss += loss.item() * imgs.size(0)

                if epoch % 5 == 0 and save:
                    # sample two images
                    for j in range(2):
                        save_img(imgs[j], 'result_imgs_val/epoch-{}-{}-original.jpg'.format(epoch, j))
                        save_img(output[j], 'result_imgs_val/epoch-{}-{}-mod.jpg'.format(epoch, j))
                    save = False

        train_loss /= len(train_data.dataset)
        val_loss /= len(val_data.dataset)

        if best_loss is None or val_loss < best_loss:
            best_loss = val_loss
            best_model_state_dict = model.state_dict()

        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time)
        s = ('(Epoch #{}) Train loss {:.4f} '
             'Val loss {:.4f} ({:.2f} s)'.format(epoch, train_loss,
                                                 val_loss, elapsed_time))
        print(s)


# size = (224, 224)
# size = (112, 112)
size = (56, 56)
dimensions = 3
means = [0.8893, 0.8280, 0.7882]
stds = [0.0828, 0.0966, 0.1033]


transformations = (
    transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # transforms.Normalize(means, stds),
    ])
)

torch.set_num_threads(4)

folder = 'imgs_partial'
model_name = 'model.bin'

# compute mean and std of images

# from PIL import Image

# folder = 'imgs_partial/train/class'
# files = os.listdir(folder)

# tensors = []
# to_tensor = transforms.ToTensor()
# resizer = transforms.Resize(size)

# for img in files:
#     img = os.path.join(folder, img)
#     im = resizer(Image.open(img))
#     t = to_tensor(im)
#     tensors.append(t)

# t = torch.stack(tensors)
# print(torch.mean(t[:, 0]), torch.std(t[:, 0]))
# print(torch.mean(t[:, 1]), torch.std(t[:, 1]))
# print(torch.mean(t[:, 2]), torch.std(t[:, 2]))
# exit()


with suppress(Exception):
    os.remove(model_name)

epochs = 200

# load data and data handler
train_dataset = datasets.ImageFolder(os.path.join(folder, 'train'), transformations)
train_data = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)

val_dataset = datasets.ImageFolder(os.path.join(folder, 'val'), transformations)
val_data = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=1)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

autoencoder = AutoEncoder()

optimiser = optim.Adam(autoencoder.parameters(), lr=0.02)
criterion = nn.MSELoss()

train(device, autoencoder, train_data, val_data, epochs=epochs,
      optimiser=optimiser, criterion=criterion, model_name=model_name)
