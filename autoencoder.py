import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms


def save_img(img, name):
    global dimensions

    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))

    if dimensions == 1:
        npimg = npimg.reshape(*size)
        npimg = np.stack((npimg,) * 3, axis=-1)
    plt.imsave(name, npimg, cmap='gray')


class PrintLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.size())
        return x


class Noiser():
    def __init__(self, noise_chance):
        self.noise_chance = noise_chance

    def __call__(self, imgs):
        for img in imgs:
            for i in range(img.size(1)):
                for j in range(img.size(2)):
                    if np.random.random() < self.noise_chance:
                        img[:, i, j] = 0


class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super().__init__()

        self.conv1 = nn.Conv2d(input_size, output_size,
                               kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(output_size)

        self.conv2 = nn.Conv2d(output_size, output_size,
                               kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(output_size)

        self.conv_transpose = nn.ConvTranspose2d(output_size, output_size,
                                                 kernel_size=2,
                                                 stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(output_size)

        self.conv_transpose_res = nn.ConvTranspose2d(input_size, output_size,
                                                     kernel_size=2,
                                                     stride=2, padding=0)
        self.bn4 = nn.BatchNorm2d(output_size)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv_transpose(out)
        out = self.bn3(out)

        out = out + self.conv_transpose_res(residual)
        out = F.relu(out)
        return out


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder
        self.encoder = nn.Sequential(*list(models.resnet50().children())[:-4],
                                     nn.Conv2d(512, 7, kernel_size=1, stride=1))

        # 7 x 7 x 7 representation

        # decoder
        self.decoder = nn.Sequential(
            ResidualBlock(7, 64, 3, 1, 1),

            ResidualBlock(64, 32, 3, 1, 1),

            ResidualBlock(32, dimensions, 3, 1, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(model, train_data, test_data, epochs, optimiser,
          criterion, scheduler, model_name='model.bin', L2_factor=0.02):
    global device

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

    noiser = Noiser(0.05)

    for epoch in range(initial_epoch, epochs):
        start_time = time.perf_counter()
        if epoch != 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimiser_state_dict': optimiser.state_dict(),
                        'best_loss': best_loss,
                        'best_model_state_dict': best_model_state_dict,
                        }, model_name)

        train_loss = 0
        with torch.set_grad_enabled(True):
            scheduler.step()
            print_error = True
            # set model for training
            model.train()

            for imgs, _ in train_data:
                imgs = imgs.to(device)

                noisy_imgs = noiser(imgs)

                output = model(noisy_imgs)

                loss = criterion(output, imgs)

                # zero gradients per batch
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                if print_error:
                    print(list(model.parameters())[0].grad)
                    print_error = False

                # compute total loss
                train_loss += loss.item() * imgs.size(0)

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
                        save_img(imgs[j], 'result_imgs/epoch-{}-{}-original.jpg'.format(epoch, j))
                        save_img(output[j], 'result_imgs/epoch-{}-{}-mod.jpg'.format(epoch, j))
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


def gen_metadataset(model, model_name, dataset_folders):
    global size
    global dimensions
    global dataset_name

    class ImageFolderWithPaths(datasets.ImageFolder):
        # override the __getitem__ method. this is the method dataloader calls
        def __getitem__(self, index):
            original_tuple = super().__getitem__(index)
            # the image file path
            path = self.imgs[index][0]
            return (path, *original_tuple)

    if dimensions == 1:
        transformations = (
            transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(size),
                transforms.ToTensor()
            ])
        )
    else:
        transformations = (
            transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor()
            ])
        )

    if model_name in os.listdir():
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['best_model_state_dict'])
    else:
        raise Exception('Model not found')

    tensors_with_paths = []
    with torch.set_grad_enabled(False):
        model.eval()
        for folder in dataset_folders:
            # load folder (train and val)
            dataset = ImageFolderWithPaths(folder, transformations)
            data = torch.utils.data.DataLoader(
                dataset, batch_size=32, shuffle=False, num_workers=4)

            for paths, imgs, _ in data:
                imgs = imgs.to(device)

                encoded_imgs = model.encoder(imgs)
                for path, ei in zip(paths, encoded_imgs):
                    # flatten tensor and append to list
                    tensors_with_paths.append(
                        (os.path.basename(path), ei.view(-1).numpy()))

    with open(dataset_name, 'w') as f:
        for path, tensor in tensors_with_paths:
            metafeatures = ','.join(str(v) for v in tensor)
            f.write('{},{}\n'.format(path, metafeatures))


# size = (224, 224)
size = (112, 112)
# size = (56, 56)
dimensions = 3

transformations = (
    transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
)


folder = 'imgs-16x16'
model_name = 'model.bin'
epochs = 50

# load data and data handler
train_dataset = datasets.ImageFolder(os.path.join(folder, 'train'), transformations)
train_data = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = datasets.ImageFolder(os.path.join(folder, 'val'), transformations)
val_data = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

autoencoder = AutoEncoder()

optimiser = optim.Adam(autoencoder.parameters(), lr=0.01, weight_decay=0.01)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)

train(autoencoder, train_data, val_data, epochs=epochs,
      optimiser=optimiser, criterion=criterion, scheduler=scheduler, model_name=model_name)
