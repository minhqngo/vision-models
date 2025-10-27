import cv2
import os
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def onehot(n_classes, target):
    vec = torch.zeros(n_classes, dtype=torch.float32)
    vec[target] = 1.
    return vec


dir_structure_help = r"""
TinyImageNetPath
├── test
│   └── images
│       ├── test_0.JPEG
│       ├── t...
│       └── ...
├── train
│   ├── n01443537
│   │   ├── images
│   │   │   ├── n01443537_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01443537_boxes.txt
│   ├── n01629819
│   │   ├── images
│   │   │   ├── n01629819_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01629819_boxes.txt
│   ├── n...
│   │   ├── images
│   │   │   ├── ...
│   │   │   └── ...
├── val
│   ├── images
│   │   ├── val_0.JPEG
│   │   ├── v...
│   │   └── ...
│   └── val_annotations.txt
├── wnids.txt
└── words.txt
"""


class TinyImagenetDataStruct:
    def __init__(self, root_dir):
        train_path = os.path.join(root_dir, 'train')
        val_path = os.path.join(root_dir, 'val')
        test_path = os.path.join(root_dir, 'test')

        wnids_path = os.path.join(root_dir, 'wnids.txt')
        words_path = os.path.join(root_dir, 'words.txt')

        self._init_data(train_path, val_path, test_path, wnids_path, words_path)

    def _init_data(self, train_path, val_path, test_path, wnids_path, words_path):
        self.ids = []
        with open(wnids_path, 'r') as idf:
            for nid in idf:
                self.ids.append(nid.strip())

        self.nid_to_words = defaultdict(list)
        with open(words_path, 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                labels = list(map(lambda x: x.strip(), labels.split(',')))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            'train': [],  # [img_path, id, nid, box]
            'val': [],  # [img_path, id, nid, box]
            'test': list(map(lambda x: os.path.join(test_path, x), os.listdir(test_path)))  # img_path
        }

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, f"{nid}_boxes.txt")
            imgs_path = os.path.join(train_path, nid, 'images')
            label_id = self.ids.index(nid)
            with open(anno_path, 'r') as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths['train'].append((fname, label_id, nid, bbox))
        self.n_classes = len(train_nids)

        # Get the validation paths and labels
        with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, 'images', fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths['val'].append((fname, label_id, nid, bbox))


class TinyImagenetDataset(Dataset):
    def __init__(self, root_dir, mode='train', transforms=None):
        paths_data = TinyImagenetDataStruct(root_dir)
        self.mode = mode
        self.transform = transforms
        self.samples = paths_data.paths[self.mode]
        self.n_classes = paths_data.n_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.mode != 'test':
            path, target, nid, box = self.samples[index]
        else:
            path = self.samples[index]
            target = None

        image = cv2.imread(path)
        if self.transform:
            image = self.transform(image)
        if target is not None:
            return image, target
        else:
            return image


def tiny_imagenet_loader(root_dir, batch_size=32):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=60),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_ds = TinyImagenetDataset(root_dir, mode='train', transforms=train_transform)
    val_ds = TinyImagenetDataset(root_dir, mode='val', transforms=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    n_classes = train_ds.n_classes

    return train_loader, val_loader, n_classes
