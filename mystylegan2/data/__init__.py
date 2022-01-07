from data.transforms import get_transform
from data.datasets import csvMultilabelDataset

def make_dataset(cfg):
    if cfg.dataname == 'cryptopunk':
        transforms = get_transform('cryptopunk')
        _dataset = csvMultilabelDataset(image_root= cfg.image_root, path_csv= cfg.datapath, transform=transforms, res= cfg.resolution)

    return _dataset


def get_data_loader(dataset, batch_size, num_workers):
    """
    generate the data_loader from the given dataset
    :param dataset: dataset for training (Should be a PyTorch dataset)
                    Make sure every item is an Image
    :param batch_size: batch size of the data
    :param num_workers: num of parallel readers
    :return: dl => data_loader for the dataset
    """
    from torch.utils.data import DataLoader

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    return dl