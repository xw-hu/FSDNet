import os
import os.path

import torch.utils.data as data
from PIL import Image


def make_dataset(root, is_train):
    if is_train:

        img_txt = open(os.path.join(root, 'train_category.txt'))

        img_name = []
        img_category = []

        for img_list in img_txt:
            x = img_list.split()
            img_name.append([os.path.join(root, x[0]), (os.path.join(root, x[1]))])
            img_category.append(x[2])

        img_txt.close()



        return img_name, img_category


    else:

        img_txt = open(os.path.join(root, 'val_category.txt'))

        img_name = []
        img_category = []

        for img_list in img_txt:
            x = img_list.split()
            img_name.append([os.path.join(root, x[0]), (os.path.join(root, x[1]))])
            img_category.append(x[2])

        img_txt.close()

        return img_name, img_category



class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None, is_train=True, batch_size=4):
        self.root = root
        self.imgs, self.imgs_category = make_dataset(root, is_train)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.batch_size = batch_size

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index % len(self.imgs)]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        img_category = self.imgs_category[index % len(self.imgs)]


        return img, target, img_category

    def __len__(self):
        return len(self.imgs) + self.batch_size - (len(self.imgs) % self.batch_size)


