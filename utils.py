import os

import torch
from torchvision import datasets, models, transforms


class ImageFolderWithPaths(datasets.ImageFolder):
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        return (path, *original_tuple)


def gen_dataset_with_deep_features(device, model, model_name,
                                   size, dimensions,
                                   dataset_name, dataset_folders):
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
