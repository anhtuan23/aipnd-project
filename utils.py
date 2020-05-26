import torch
from torchvision import models
from torch import optim, nn

TESTED_MODELS = {'vgg16_bn' : {'load_command': 'models.vgg16_bn(pretrained=True)',
                               'classifier_input': 25088
                              },
                 'densenet121' : {'load_command': 'models.densenet121(pretrained=True)',
                                  'classifier_input': 1024
                                 }
                }

def get_model(arch):
    if arch not in TESTED_MODELS:
        raise Exception(f'Sorry, arch {arch} is not supported. Please use either: {TESTED_MODELS.keys()}')
    return TESTED_MODELS[arch]


def load_model(arch, device, output_num, hidden_units_num):
        
    # Load model
    model = eval(get_model(arch)['load_command'])

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # new classifier for model
    model.classifier = nn.Sequential(nn.Linear(get_model(arch)['classifier_input'], hidden_units_num),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(hidden_units_num, output_num),
                                     nn.LogSoftmax(dim=1))
    
    model.to(device)
    
    return model


def get_optimizer(model, learn_rate, state_dict=None):
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    if state_dict != None:
        optimizer.load_state_dict(state_dict)
    return optimizer


def get_device(use_gpu):
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print('GPU is not available. Using CPU instead.')
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    
    print("Operate in:", device)
    return device