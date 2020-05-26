import torch
from torchvision import datasets, transforms
import json

TRAIN = 'train'
VALID = 'valid'
TEST = 'test'
DATA_CATS = [TRAIN, VALID, TEST]

CLASS_NUM = 102

def get_data(data_dir):
    data_transforms = {
        TRAIN: transforms.Compose([transforms.RandomRotation(30),
                                   transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])
                                  ]),
        VALID: transforms.Compose([transforms.Resize(255),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])
                                   ]),
        TEST: transforms.Compose([transforms.Resize(255),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])
                                 ])
    }


    # Load the datasets with ImageFolder
    image_datasets = {cat : datasets.ImageFolder(data_dir + '/' + cat, transform=data_transforms[cat]) for cat in DATA_CATS}


    # Using the image datasets and the trainforms, define the dataloaders
    data_loaders = {cat : torch.utils.data.DataLoader(image_datasets[cat], batch_size=64, shuffle=cat==TRAIN) 
                    for cat in DATA_CATS}
    
    for cat in DATA_CATS:
        print(f'Loaded {len(image_datasets[cat])} images under {cat}')
        
    return (data_loaders, image_datasets)


def get_cat_to_name(category_names_path):
    with open(category_names_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name