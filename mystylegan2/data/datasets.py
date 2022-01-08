from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import os

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

attcol = ['Male', 'Female', 'Green Eye Shadow', 'Earring', 'Blonde Bob', 'Smile', 'Mohawk', 
'Wild Hair', 'Nerd Glasses', 'Pipe', 'Big Shades', 'Goat', 'Purple Eye Shadow', 'Half Shaved',
'Do-rag', 'Wild White Hair', 'Spots', 'Clown Eyes Blue', 'Messy Hair', 
'Luxurious Beard', 'Clown Nose', 'Police Cap', 'Big Beard', 'Blue Eye Shadow',
'Black Lipstick', 'Clown Eyes Green', 'Straight Hair Dark', 'Blonde Short',
'Purple Lipstick', 'Straight Hair Blonde', 'Pilot Helmet', 'Hot Lipstick', 
'Regular Shades', 'Stringy Hair', 'Small Shades', 'Frown', 'Eye Mask', 'Muttonchops',
'Bandana', 'Horned Rim Glasses', 'Crazy Hair', 'Classic Shades', 'Handlebars', 'Mohawk Dark', 
'Dark Hair', 'Peak Spike', 'Normal Beard Black', 'Cap', 'VR', 'Frumpy Hair', 'Cigarette', 'Normal Beard',
'Red Mohawk', 'Shaved Head', 'Chinstrap', 'Mole', 'Knitted Cap', 
'Fedora', 'Shadow Beard', 'Straight Hair', 'Hoodie', 'Eye Patch', 'Headband', 'Cowboy',
'Hat', 'Tassle Hat', '3D Glasses', 'Mustache', 'Vape', 'Choker', 'Pink With Hat', 
'Welding Goggles', 'Vampire Hair', 'Mohawk Thin', 'Tiara', 'Front Beard Dark', 
'Cap Forward', 'Gold Chain', 'Purple Hair', 'Beanie', 'Clown Hair Green', 'Pigtails', 
'Silver Chain', 'Front Beard', 'Rosy Cheeks', 'Orange Side', 'Wild Blonde', 'Buck Teeth', 
'Top Hat', 'Medical Mask', ' Medium', ' Dark', ' Light', ' Albino']

def getAttributeIndex(attname):
    res = []
    for i in Attributes[attname]:
        res.append((i, attcol.index(i)))
    return res

def getListAttributes(csvpath, start_index= 1):
    df = pd.read_csv(csvpath)
    return list(df.columns)[start_index:]

class csvMultilabelDataset(Dataset):
    def __init__(self, image_root, path_csv, transform= None, target_transform= None, res= 0):
        df = pd.read_csv(path_csv)
        self.samples = df['path']
        self.labels = df.drop(['Unnamed: 0','path'], axis= 1)
        self.transform = transform
        self.target_transform = target_transform
        self.list_attributes = list(df.columns)[2:]
        self.image_root =image_root
        self.res = res if res != 0 else 32

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = os.path.join(self.image_root, self.samples[index])
        label = np.asarray(list(self.labels.loc[index]), dtype= np.float32)
        try:
            image = Image.open(path)
            fimage = Image.new('RGBA', image.size, (102, 133, 151, 255))
            fimage.paste(image, (0, 0), mask=image)
            fimage = fimage.convert("RGB")
        except:
            raise Exception('Can\'t load image')

        right = 4
        left = 4
        top = 4
        bottom = 4
          
        width, height = image.size
          
        new_width = width + right + left
        new_height = height + top + bottom
          
        result = Image.new(fimage.mode, (new_width, new_height), (102, 133, 151))
          
        result.paste(fimage, (left, top))
        result = result.resize((self.res, self.res), 0)

        if self.transform:
            result = self.transform(result)
        if self.target_transform:
            label = self.target_transform(label)
        
        return result, label
