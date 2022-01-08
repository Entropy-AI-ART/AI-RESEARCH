import numpy as np
import PIL.Image
from data.datasets import Attributes

def randomAttribute(list_attributes, max_count= 5):
    count = np.random.randint(max_count, size= 1)
    gender = np.random.choice(Attributes['Gender'], size= 1)
    # gender = Attributes['Gender'][gender]
    skintone = np.random.choice(Attributes['Skin'], size= 1)
    # skintone = Attributes['Skin'][skintone]
    mainAtts = [k for k, _ in Attributes.items()][2:]
    
    chosen_mainatt = np.random.choice(mainAtts, size= count)

    att_persam = []
    for att in chosen_mainatt:
        att_persam.append(np.random.choice(Attributes[att], size= 1))
    
    sum_atts = list(gender) + list(skintone) + att_persam
    label = []
    for att in list_attributes:
        if att not in sum_atts:
            label.append(0)
        else:
            label.append(1)

    return np.asarray(label)

def get_batchRandomAttribute(list_attributes, batch_size, max_count= 5):
    return np.stack([randomAttribute(list_attributes, max_count) for _ in range(batch_size)])