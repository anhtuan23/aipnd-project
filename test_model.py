import torch
from torch import nn

def test_model_accuracy(model_, testloader_, device_):
    model_.to(device_)
    
    criterion = nn.NLLLoss()
    
    test_loss = 0
    accuracy = 0

    model_.eval()

    with torch.no_grad():
        for inputs, labels in testloader_:
            inputs, labels = inputs.to(device_), labels.to(device_)

            log_ps = model_(inputs)
            batch_loss = criterion(log_ps, labels)

            test_loss += batch_loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f'Test Loss: {test_loss/len(testloader_):.3f}',
         f'Valid Accuracy: {accuracy/len(testloader_):.3f}')