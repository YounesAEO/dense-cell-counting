import json
import argparse
import os, os.path
import PIL.Image as Image
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(description='Make Dataset')

parser.add_argument('train_folder', metavar='TRAIN',
                    help='path to train directory')

def make_lists(train_path, is_json=False):
    if is_json:
        with open(os.path.join(train_path, "training.json"), 'r') as json_train:
            data = json.load(json_train)
        images_list = [element["image"]["pathname"][1:] for element in data]
    else:
        images_list = [name for name in os.listdir(os.path.join(train_path,"images")) if os.path.isfile(os.path.join(train_path,"images", name))]
    ## get the size of the folder
    images_len = len(images_list)
    train_list = [os.path.join(os.path.abspath(train_path), img) for img in images_list[:int(0.8*images_len)]]
    val_list = [os.path.join(os.path.abspath(train_path), img) for img in images_list[int(0.8*images_len):]]

    return train_list, val_list

## for test purposes
if __name__=='__main__':
    global args
    args = parser.parse_args()
    train_list, val_list = make_lists(args.train_folder, is_json=True)
    
    with open('train.json', 'w') as f:
        json.dump(train_list, f)

    with open('val.json', 'w') as f:
        json.dump(train_list, f)
