import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
import json
from get_input_args import get_predict_input_args

def load(path, gpu=False):
    if gpu:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc:storage)
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])
    epochs = checkpoint['epoch']
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0)
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return epochs, model, optimizer

def process_image(image, gpu=False):
    # Resize and crop
    image = image.resize((255, 255))
    image = image.crop((16, 16, 240, 240))
    
    # Convert to np array and normalize all values to between 0 and 1
    np_image = np.array(image)
    np_image = np_image/255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Preprocess the image
    np_image = (np_image - mean) / std
    # Torch wants color first, but it's the third dimension in the numpy array
    np_image = np_image.transpose((2, 0, 1))
    # Convert the image to a tensor
    if gpu:
        image_tensor = torch.from_numpy(np_image).type(torch.cuda.FloatTensor)
    else:
        image_tensor = torch.from_numpy(np_image).type(torch.FloatTensor)
    return image_tensor

def predict(image_path, model, topk=5, gpu=False):
   
    processed_image = process_image(Image.open(image_path), gpu)
    processed_image = processed_image.unsqueeze_(0)
    probs = torch.exp(model.forward(processed_image))
    top_p, top_class = probs.topk(topk, dim=1)
    return top_p, top_class

def main():
    in_arg = get_predict_input_args()
    epochs, loaded_model, optimizer = load(in_arg.checkpoint, in_arg.gpu)
    loaded_model.to("cuda" if in_arg.gpu else "cpu")
    probs, classes = predict(in_arg.path_to_image, loaded_model, in_arg.top_k, in_arg.gpu)
    classes = np.array(classes)
    
    # Load the class to name json
    cat_to_name = dict()
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    # Reverse class to idx dictionary
    idx_to_class = {val: key for key, val in loaded_model.class_to_idx.items()}
    
    # Convert numbers to classes
    adjusted_classes = [idx_to_class[ndx] for ndx in classes[0]]
    
    # Convert classes to flowers
    flowers = [cat_to_name[ndx] for ndx in adjusted_classes]
    flower_probs = dict()
    for i in range(in_arg.top_k):
        flower_probs[flowers[i]] = "{:.3f}%".format(probs.cpu().numpy()[0][i])
    print("Predictions for image {}".format(in_arg.path_to_image))
    print("-----" * 5)
    for key in flower_probs:
        print("{}: {}".format(key, flower_probs.get(key)))

if __name__ == "__main__":
    main()



