# Argparsers
import argparse

def get_train_input_args():
    parser = argparse.ArgumentParser(description='enter file data')
    
    parser.add_argument('data_dir', type=str, default='flowers', help='path to image directories')
    parser.add_argument('--arch', type=str, default='vgg16', help='torch model architecture')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='where to save the trained model')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=.001, help='learning rate of backprop algorithm')
    parser.add_argument('--hidden_units', default=[4096, 1024], nargs=2, metavar=('hidden_input', 'hidden_output'),
                        type=int, help='dimensions of hidden layer')
    parser.add_argument('--gpu', action='store_true')
    return parser.parse_args()

def get_predict_input_args():
    parser = argparse.ArgumentParser(description='enter file data')
    
    parser.add_argument('path_to_image', type=str, default='', help='path to image')
    parser.add_argument('checkpoint', type=str, default='checkpoint.pth', help='path to model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='return k most probable results')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true')
    return parser.parse_args()
    
    
    