import argparse

import torch
from image_preprocessing import process_image
from utils import get_device
from checkpoint_utils import load_checkpoint
from load_data import get_cat_to_name

parser = argparse.ArgumentParser()
parser.add_argument('image_path', action="store")
parser.add_argument('checkpoint_path', action="store")
parser.add_argument('--top_k', action='store', dest='top_k', default=1, type=int)
parser.add_argument('--category_names', action='store', dest='category_names', default ='cat_to_name.json')
parser.add_argument('--gpu', action='store_true', default=False, dest='gpu')

args = parser.parse_args()
print("Image path:", args.image_path)
print("Checkpoint path:", args.checkpoint_path)
print("Top k:", args.top_k)
print("Category names:", args.category_names)
print("Use GPU:", args.gpu)

def predict(image_path, model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    np_img = process_image(image_path)
    image = torch.from_numpy(np_img).type(torch.FloatTensor)
    image.unsqueeze_(0) 
    
    image = image.to(device)
    
    ps = torch.exp(model(image))
                   
    top_p, top_idx = ps.topk(topk, dim=1)
    
    #convert tensor to list
    top_p = top_p.tolist()[0]
    top_idx = top_idx.tolist()[0]
    
    # get idx_to_class dictionary
    idx_to_class = {idx: klass for klass, idx in model.class_to_idx.items()} 
    
    #convert top_idx to top_class (start from 1 like in directory structure)
    top_class = [idx_to_class[idx] for idx in top_idx]
    
    return (top_p, top_class)


print('Loading model from checkpoint ...')
device = get_device(args.gpu)
model, optimizer = load_checkpoint(args.checkpoint_path, device)

# Try to predict an image
probs, id_classes = predict(args.image_path, model, device, args.top_k)

# convert id class to name class
cat_to_name = get_cat_to_name(args.category_names)
name_classes = [cat_to_name[num_class] for num_class in id_classes]

print('\nPredictions:')
for p, name in zip(probs, name_classes):
    print(f'\t{name.capitalize()} \t: {p*100:0.2f}%')

