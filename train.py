import argparse

import torch
from torch import nn

from workspace_utils import keep_awake

from utils import get_device, load_model, get_optimizer
from load_data import get_data, CLASS_NUM, TRAIN, VALID, TEST
from checkpoint_utils import save_checkpoint
from test_model import test_model_accuracy

# Argument parsers
parser = argparse.ArgumentParser()
parser.add_argument('data_directory', action="store")
parser.add_argument('--save_dir', action='store', dest='save_dir')
parser.add_argument('--arch', action='store', default='vgg16_bn', dest='arch')
parser.add_argument('--learning_rate', action='store', default=0.003, dest='learning_rate', type=float)
parser.add_argument('--hidden_units ', action='store', default=1024, dest='hidden_units', type=int)
parser.add_argument('--epochs', action='store', default=2, dest='epochs', type=int)
parser.add_argument('--gpu', action='store_true', default=False, dest='gpu')

args = parser.parse_args()
print("Data dir:", args.data_directory)
print("Save dir:", args.save_dir)
print("Arch:", args.arch)
print("Learning rate:", args.learning_rate)
print("Hidden units:", args.hidden_units)
print("Epochs:", args.epochs)
print("Use GPU:", args.gpu)


def train_model(model, trainloader, validloader, device, optimizer, epochs):
    criterion = nn.NLLLoss()

    step = 0
    print_every = 5

    train_loss = 0

    for e in keep_awake(range(epochs)):

        for inputs, labels in trainloader:
            step += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if step % print_every == 0:
                valid_loss = 0
                accuracy = 0

                model.eval()

                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        log_ps = model(inputs)
                        batch_loss = criterion(log_ps, labels)

                        valid_loss += batch_loss.item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                model.train()

                print(f'Epoch: {e + 1}/{epochs}',
                     f'Training Loss: {train_loss/print_every:.3f}',
                     f'Valid Loss: {valid_loss/len(validloader):.3f}',
                     f'Valid Accuracy: {accuracy/len(validloader):.3f}')
                train_loss = 0
    return model


device = get_device(args.gpu)
data_loaders, img_datasets = get_data(args.data_directory)

print('Training new model ...')
model = load_model(args.arch, device, CLASS_NUM, args.hidden_units)
optimizer = get_optimizer(model, args.learning_rate)
model = train_model(model, data_loaders[TRAIN], data_loaders[VALID], device, optimizer, args.epochs)
save_checkpoint(model, img_datasets[TRAIN], optimizer, args.arch, CLASS_NUM, args.learning_rate, args.epochs, args.hidden_units)
print("Testing Model accuracy ...")
test_model_accuracy(model, data_loaders[TEST], device)
