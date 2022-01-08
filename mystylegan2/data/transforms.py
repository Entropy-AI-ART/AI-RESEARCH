def get_transform(dataname, resize= None):
    """
    obtain the image transforms required for the input data
    :param new_size: size of the resized images
    :return: image_transform => transform object from TorchVision
    """
    # mean=[0.485, 0.456, 0.406]
    # std=[0.229, 0.224, 0.225]

    mean=[0.5, 0.5, 0.5]
    std=[0.5, 0.5, 0.5]

    from torchvision.transforms import ToTensor, Normalize, Compose, Pad, Resize
    from PIL.Image import NEAREST
    if dataname == 'cryptopunk':
        if resize:
            image_transform = Compose([Pad(4),
                            Resize(resize, NEAREST),
                            ToTensor(),
                            Normalize(mean, std)])
        else:
            image_transform = Compose([
                            ToTensor()])

    return image_transform