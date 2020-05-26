import torch
from torch import nn
from utils import get_model
from torchvision import models 
from utils import get_optimizer

def save_checkpoint(model, train_dataset, optimizer, arch, class_num, learn_rate, epochs, hidden_unit_num):
    # Save the checkpoint 
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {'model': arch,
                  'output_size': class_num,
                  'classifier': model.classifier,
                  'learnrate': learn_rate,  # should get from optimizer later 
                  'epochs' : epochs,
                  'hidden_units' : hidden_unit_num,  
                  'class_to_idx': model.class_to_idx,
                  'device' : str(next(model.parameters()).device),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'model_state_dict': model.state_dict()
                 }

    torch.save(checkpoint, arch + '_checkpoint.pth')
    
    
def load_checkpoint(filepath, device):
    
    # load on model: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices
    if device == torch.device('cuda'):
        checkpoint = torch.load(filepath)
    else:  # load on CPU
        checkpoint = torch.load(filepath, map_location='cpu')
    
    arch = checkpoint['model']
    # Load model
    model = eval(get_model(arch)['load_command'])

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # new classifier for model
    model.classifier = nn.Sequential(nn.Linear(get_model(arch)['classifier_input'], checkpoint['hidden_units']),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(checkpoint['hidden_units'], checkpoint['output_size']),
                                     nn.LogSoftmax(dim=1))
        
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    if device == torch.device('cuda'):
        model.to(device)
    
    model.eval() # set to eval mode for inference
    
    optimizer = get_optimizer(model, checkpoint['learnrate'], checkpoint['optimizer_state_dict'])
    
    return (model, optimizer)