import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from get_input_args import get_train_input_args

def main():
    
    in_arg = get_train_input_args()

    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    # Data transformation
    data_transforms = dict()
    data_transforms['training'] = transforms.Compose([transforms.RandomRotation(30),
                                                 transforms.RandomResizedCrop(224),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    data_transforms['validation'] = transforms.Compose([transforms.Resize(255),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                       [0.229, 0.224, 0.225])])

    data_transforms['testing'] = transforms.Compose([transforms.Resize(255),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                       [0.229, 0.224, 0.225])])


# Dataset ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    image_datasets = dict()
    image_datasets['training'] = datasets.ImageFolder(train_dir, transform = data_transforms['training'])
    image_datasets['validation'] = datasets.ImageFolder(valid_dir, transform = data_transforms['validation'])
    image_datasets['testing'] = datasets.ImageFolder(test_dir, transform = data_transforms['testing'])

# Dataloader ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dataloaders = dict()
    dataloaders['training'] = torch.utils.data.DataLoader(image_datasets['training'], batch_size = 16, shuffle =   True)
    dataloaders['validation'] = torch.utils.data.DataLoader(image_datasets['validation'], batch_size = 16, shuffle = False)
    dataloaders['testing'] = torch.utils.data.DataLoader(image_datasets['testing'], batch_size = 16, shuffle = False)

# Set up the model criteria
    device = torch.device("cuda" if in_arg.gpu else "cpu")

    model = models.__dict__[in_arg.arch](pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(25088, in_arg.hidden_units[0]),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(in_arg.hidden_units[0], in_arg.hidden_units[1]),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(in_arg.hidden_units[1], 102),
                           nn.LogSoftmax(dim=1)
                           )    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = in_arg.learning_rate)
    model.to(device)
    
    # Train model
    epochs = in_arg.epochs
    steps = 0
    print_every = 100
    running_loss = 0

    for epoch in range(epochs):
        running_loss = 0
        for images, labels in dataloaders['training']:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logps = model.forward(images)       
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in dataloaders['validation']:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        test_loss += criterion(logps, labels)

                        ps = torch.exp(logps)

                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))



                print("Epoch: {}/{}".format(epoch+1, epochs),
             "Training Loss: {:.3f}".format(running_loss / print_every),
             "Validation Loss: {:.3f}".format(test_loss / len(dataloaders['validation'])),
             "Validation Accuracy: {:.3f}".format(accuracy / len(dataloaders['validation'])))
                model.train()
                running_loss = 0



    # Save trained model
    torch.save({
        'epoch': epochs,
        'arch': in_arg.arch,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': image_datasets['training'].class_to_idx,
        'classifier': model.classifier
    }, in_arg.save_dir
    )
    
if __name__ == "__main__":
    main()

