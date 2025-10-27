DATASET_STATS = {
    'cifar': {
        'mean': (0.4915, 0.4823, 0.4468),
        'std': (0.2470, 0.2435, 0.2616)
    },

    'svhn': {
        'mean': (0.4377, 0.4438, 0.4728),
        'std': (0.1980, 0.2010, 0.1970)
    }
}


def torch_dataset_factory(dataset_name, dataset_root, bs=32):
    from torchvision import datasets, transforms
    from torch.utils import data

    if 'cifar' in dataset_name.lower():
        mean = DATASET_STATS['cifar']['mean']
        std = DATASET_STATS['cifar']['std']
    else:  # svhn dataset
        mean = DATASET_STATS['svhn']['mean']
        std = DATASET_STATS['svhn']['std']

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])

    if dataset_name.lower() == 'cifar10':
        train_dataset = datasets.CIFAR10(root=dataset_root,
                                         train=True,
                                         download=True,
                                         transform=train_transform)
        test_dataset = datasets.CIFAR10(root=dataset_root,
                                        train=False,
                                        download=True,
                                        transform=test_transform)
        n_classes = len(train_dataset.classes)

    elif dataset_name.lower() == 'cifar100':
        train_dataset = datasets.CIFAR100(root=dataset_root,
                                          train=True,
                                          download=True,
                                          transform=train_transform)
        test_dataset = datasets.CIFAR100(root=dataset_root,
                                         train=False,
                                         download=True,
                                         transform=test_transform)
        n_classes = len(train_dataset.classes)
        
    elif dataset_name.lower() == 'svhn':
        train_dataset = datasets.SVHN(root=dataset_root,
                                      split='train',
                                      download=True,
                                      transform=train_transform)
        test_dataset = datasets.SVHN(root=dataset_root,
                                     split='test',
                                     download=True,
                                     transform=test_transform)
        n_classes = 10

    else:
        raise NotImplementedError(f"Image classification for dataset {dataset_name} is not impldmented")

    train_loader = data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=2)
    test_loader = data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=2)
    return train_loader, test_loader, n_classes
