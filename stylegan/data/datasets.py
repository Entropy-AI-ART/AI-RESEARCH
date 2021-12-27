import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import pandas as pd

Attributes = {'Gender' : ['Male', 'Female'],
             'Skin' : ['Medium', 'Light', 'Dark', 'Albino'],
             'Hair' : ['Bandana', 'Beanie', 'Blonde Bob', 'Blonde Short',
                       'Cap', 'Cap Forward', 'Clown Hair Green', 'Cowboy Hat',
                       'Crazy Hair', 'Dark Hair', 'Do-rag', 'Fedora',
                       'Frumpy Hair', 'Half Shaved', 'Headband', 'Hoodie',
                       'Knitted Cap', 'Messy Hair', 'Mohawk', 'Mohawk Dark',
                        'Mohawk Thin','Orange Side', 'Peak Spike', 'Pigtails',
                        'Pilot Helmet', 'Pink With Hat', 'Police Cap', 'Purple Hair', 'Red Mohawk',
                        'Shaved Head', 'Straight Hair', 'Straight Hair Blonde', 'Straight Hair Dark',
                       'Stringy Hair', 'Tassle Hat', 'Tiara', 'Top Hat',
                       'Wild Blonde', 'Wild Hair', 'Wild White Hair', 'Vampire Hair'],
             'Eyes' : ['3D Glasses', 'Big Shades', 'Blue Eye Shadow','Classic Shades',
                       'Clown Eyes Blue', 'Clown Eyes Green', 'Eye Mask', 'Eye Patch',
                       'Green Eye Shadow', 'Horned Rim Glasses', 'Nerd Glasses', 'Purple Eye Shadow',
                       'Regular Shades', 'Small Shades', 'VR', 'Welding Goggles'],
             'Facial Hair' : ['Chinstrap', 'Big Beard', 'Front Beard', 'Front Beard Dark',
                              'Goat', 'Handlebars', 'Luxurious Beard', 'Mustache',
                              'Muttonchops', 'Normal Beard', 'Normal Beard Black', 'Shadow Beard'],
             'Neck Accessory' : ['Choker', 'Gold Chain', 'Silver Chain'],
             'Mouth Prop' : ['Cigarette', 'Medical Mask', 'Pipe', 'Vape'],
             'Mouth' : ['Black Lipstick', 'Buck Teeth', 'Frown', 'Hot Lipstick',
                        'Purple Lipstick', 'Smile'],
             'Blemishes' : ['Mole', 'Rosy Cheeks', 'Spots'],
             'Ears' : ['Earring'],
             'Nose' : ['Clown Nose']}

def getListAttributes(csvpath, start_index= 1):
    df = pd.read_csv(csvpath)
    return list(df.columns)[start_index:]

class csvMultilabelDataset(Dataset):
    def __init__(self, path_csv, transform= None, target_transform= None):
        df = pd.read_csv(path_csv)
        self.samples = df['path']
        self.labels = df.drop(['Unnamed: 0','path'], axis= 1)
        self.transform = transform
        self.target_transform = target_transform
        self.list_attributes = list(df.columns)[1:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        label = list(self.labels.loc[index])
        try:
            image = Image.open(path).convert("RGB")
        except:
            raise Exception('Can\'t load image')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
