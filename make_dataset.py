import argparse
from re import X
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import scipy.spatial as sp
import os
import glob
from scipy.ndimage import gaussian_filter 
import scipy
import json
import errno
import xml.etree.ElementTree as ET


parser = argparse.ArgumentParser(description='Make Dataset')

parser.add_argument('train_folder', metavar='TRAIN',
                    help='path to train directory')
parser.add_argument('gt_format', metavar='GT_FORMAT',
                    help='annotation files format (xml, json, mat)')
parser.add_argument('--test_folder','-t', metavar='TEST',
                    help='path to test directory')
parser.add_argument('--format', '-f', metavar='FORMAT',
                    help='dataset images format (png, jpg, jpeg)')
parser.add_argument('--method', '-m', metavar='METHOD',
                    help='Method of density generation\n g: geometric adaptive \n f: fixed kernel \n default is -f')
parser.add_argument('--sigma', '-s', metavar='SIGMA', default=15,
                    help='standard deviation of gaussian kernel if the fixed option was used')

def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = sp.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2.
        density += gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density

def build_image_paths(path_sets):
    img_paths = [] 
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, f'*.{args.format}')):
            img_paths.append(img_path)
    return img_paths

def loadgt(path, color_list= [-16711706, -16711681]):
    if args.gt_format == "mat":
        mat = io.loadmat(path)
        return mat["image_info"][0,0][0,0][0]
    elif args.gt_format == "xml":
        tree = ET.parse(path)
        parent_tag = tree.getroot()
        pos_x = parent_tag.findall("./rois/roi/position/pos_x")
        pos_y = parent_tag.findall("./rois/roi/position/pos_y")
        colors = parent_tag.findall("./rois/roi/color")
        gt = []
        for x,y,c in zip(pos_x,pos_y, colors):
            if int(c.text) in color_list:
                gt.append([float(x.text),float(y.text)])
        return np.array(gt)

def loadgt_from_json(path, image_infos):
    return np.array(image_infos[path])

def get_annotations_from_bb():
    ## get the training file
    with open(os.path.join(args.train_folder, "training.json"), 'r') as json_train:
        data = json.load(json_train)
    image_infos = {}
    for element in data:
        ## metadata : checksum, pathname, shape (r,c,channels) ?
        image_pathname = element['image']["pathname"][1:]
        image_objects = element['objects']
        image_infos[image_pathname] = []
        for obj in image_objects:
            cmax = obj["bounding_box"]["maximum"]["c"]
            cmin = obj["bounding_box"]["minimum"]["c"]
            rmax = obj["bounding_box"]["maximum"]["r"]
            rmin = obj["bounding_box"]["minimum"]["r"]
            # (cmin, rmin) == (x,y)
            width = cmax - cmin
            height = rmax - rmin
            centerx = cmin + width/2
            centery = rmin + height/2
            image_infos[image_pathname].append((centerx,centery))
    return image_infos
    
def main():
    global args
    args = parser.parse_args()
    path_sets = []
    
    ##check if the folders are valid
    ## we supose that the folders contain an 'images' folder and a 'ground_truth' folder
    train_images = os.path.join(args.train_folder,'images')

    if not os.path.exists(train_images):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), train_images)
    
    path_sets.append(train_images)

    if args.test_folder and os.path.exists(os.path.join(args.test_folder,'images')):
        path_sets.append(os.path.join(args.test_folder,'images'))
    
    ##check the format of the images else throw error
    args.format = "png" # FIX ME
    #args.gt_format = "xml"
    if args.gt_format not in ["json", "xml", "mat"]:
        raise ValueError("Unrecognized ground truth format (supported formats : xml, json, mat)")

    if args.gt_format == "json":
        ## create dictionary : {image_path: 2D array of annotations}
        image_infos = get_annotations_from_bb()
        ## test on image /images/8d02117d-6c71-4e47-b50a-6cc8d5eb1d55.png
        # plt.imshow(Image.open(args.train_folder + '/images/8d02117d-6c71-4e47-b50a-6cc8d5eb1d55.png'))
        # annotations = image_infos['/images/8d02117d-6c71-4e47-b50a-6cc8d5eb1d55.png']
        # xy = list(zip(*annotations))
        # plt.scatter(xy[0], xy[1])
        # plt.show()

    try:
        args.sigma = float(args.sigma)
    except:
        args.sigma = 15
    
    img_paths = build_image_paths(path_sets)
    #part = [img_paths[i] for i in range(5)]
    
    if args.method =="g":
        print("Using gerometric adaptive method \n")
    else:
        print(f'Using a gaussian kernel with sigma = {args.sigma}')
    
    for img_path in img_paths:
        print(img_path)
        img= plt.imread(img_path)
        k = np.zeros((img.shape[0],img.shape[1]))
        if args.gt_format == "json":
            gt = loadgt_from_json(img_path.replace(args.train_folder,"").replace("\\", "/"), image_infos)
        else:
            gt = loadgt(img_path.replace(f'.{args.format}',f'.{args.gt_format}').replace('images','ground_truth')) 
        for i in range(0,len(gt)):
            if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
                k[int(gt[i][1]),int(gt[i][0])]=1
        k = gaussian_filter_density(k) if args.method == 'g' else gaussian_filter(k,args.sigma)
        with h5py.File(img_path.replace(f'.{args.format}','.h5').replace('images','ground_truth'), 'w') as hf:
                hf['density'] = k

    ## for test purposes
    gt_file = h5py.File(img_paths[0].replace(f'.{args.format}','.h5').replace('images','ground_truth'),'r')
    groundtruth = np.asarray(gt_file['density'])
    plt.imshow(groundtruth,cmap=CM.jet)
    plt.show()

if __name__=="__main__":
    main()
