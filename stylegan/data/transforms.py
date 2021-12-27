def get_transform(dataname):
    """
    obtain the image transforms required for the input data
    :param new_size: size of the resized images
    :return: image_transform => transform object from TorchVision
    """
    from torchvision.transforms import ToTensor, Normalize, Compose, Pad
    if dataname == 'cryptopunk':
        image_transform = Compose([Pad(4),
                            ToTensor(),
                            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    return image_transform