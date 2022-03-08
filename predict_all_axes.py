##### MODEL AND DATA LOADING
import time

start = time.time()
import torch
# torch.cuda.empty_cache()
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import re

import os
import copy
import subprocess

from helpers import makedir, find_high_activation_crop
import model
import push
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
parser.add_argument('-partid', nargs=1, type=str)
parser.add_argument('-imgdir', nargs=1, type=str)
parser.add_argument('-masks', nargs=1, type=str, default='0')
parser.add_argument('-slicefile', nargs=1, type=str, default=None)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(args.gpuid[0])
# specify the test image to be analyzed
test_image_dir = args.imgdir[0]
x_test_image_names = []
y_test_image_names = []
z_test_image_names = []
for root, dirs, files in os.walk(os.path.join(test_image_dir, "x")):
    for filename in files:
        x_test_image_names.append(filename)
x_test_image_names = [f for f in x_test_image_names if "png" in f]
for root, dirs, files in os.walk(os.path.join(test_image_dir, "y")):
    for filename in files:
        y_test_image_names.append(filename)
y_test_image_names = [f for f in y_test_image_names if "png" in f]
for root, dirs, files in os.walk(os.path.join(test_image_dir, "z")):
    for filename in files:
        z_test_image_names.append(filename)
z_test_image_names = [f for f in z_test_image_names if "png" in f]

test_image_paths = [os.path.join(test_image_dir, "x", test_image_name) for test_image_name in x_test_image_names] + \
                [os.path.join(test_image_dir, "y", test_image_name) for test_image_name in y_test_image_names] + \
                [os.path.join(test_image_dir, "z", test_image_name) for test_image_name in z_test_image_names]
slice_numbers = [(test_image_name.split("slice")[-1]).split(".png")[0] for test_image_name in x_test_image_names] + \
                [(test_image_name.split("slice")[-1]).split(".png")[0] for test_image_name in y_test_image_names] + \
                [(test_image_name.split("slice")[-1]).split(".png")[0] for test_image_name in z_test_image_names]
slice_numbers = [int(s) for s in slice_numbers]
axis_table = ["x"]*len(x_test_image_names) + ["y"]*len(y_test_image_names) + ["z"]*len(z_test_image_names)

if args.slicefile is not None:
    slicefile = args.slicefile[0]
    good_slices = pd.read_csv(slicefile, header=None)[0].tolist()
    print(good_slices)
    test_image_paths = [test_image_paths[i] for i in range(len(test_image_paths)) if slice_numbers[i] in good_slices]

# load the model
check_test_accu = False

load_model_dir = args.modeldir[0] #'./saved_models/vgg19/003/'
load_model_name = args.model[0] #'10_18push0.7822.pth'

# participant identifier
partid = args.partid[0]

# predict masks
predict_masks = bool(int(args.masks[0]))

model_base_architecture = load_model_dir.split('/')[2]
experiment_run = '/'.join(load_model_dir.split('/')[3:])

# creating the path to save results
save_analysis_path = os.path.join(test_image_dir)
makedir(save_analysis_path)

# creating log to save results
log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'global_analysis.log'))

load_model_path = os.path.join(load_model_dir, load_model_name)
epoch_number_str = re.search(r'\d+', load_model_name).group(0)
start_epoch_number = int(epoch_number_str)

log('load model from ' + load_model_path)
log('model base architecture: ' + model_base_architecture)
log('experiment run: ' + experiment_run)

ppnet = torch.load(load_model_path)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

img_size = ppnet_multi.module.img_size
prototype_shape = ppnet.prototype_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

class_specific = True

normalize = transforms.Normalize(mean=mean,
                                 std=std)

##### SANITY CHECK
# confirm prototype class identity
load_img_dir = os.path.join(load_model_dir, 'img')

prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+epoch_number_str, 'bb'+epoch_number_str+'.npy'))
prototype_img_identity = prototype_info[:, -1]

log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
log('Their class identities are: ' + str(prototype_img_identity))

# confirm prototype connects most strongly to its own class
prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
prototype_max_connection = prototype_max_connection.cpu().numpy()
if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
    log('All prototypes connect most strongly to their respective classes.')
else:
    log('WARNING: Not all prototypes connect most strongly to their respective classes.')

# load the test image and forward it through the network
preprocess = transforms.Compose([
   transforms.Resize((img_size,img_size)),
   transforms.ToTensor(),
   normalize
])

# loop here over all image paths
tables = []
dct_prototype_activations = {}
dct_prototype_activation_patterns = {}
for k, test_image_path in enumerate(test_image_paths):
    img_pil = Image.open(test_image_path).convert('RGB')
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    images_test = img_variable.cuda()
    logits, min_distances = ppnet_multi(images_test)
    conv_output, distances = ppnet.push_forward(images_test)
    # logits, min_distances = ppnet_multi(img_variable)
    # conv_output, distances = ppnet.push_forward(img_variable)
    prototype_activations = ppnet.distance_2_similarity(min_distances)
    prototype_activation_patterns = ppnet.distance_2_similarity(distances)
    if ppnet.prototype_activation_function == 'linear':
        prototype_activations = prototype_activations + max_dist
        prototype_activation_patterns = prototype_activation_patterns + max_dist
    # save each prediction and corresponding slice number in tables
    for i in range(logits.size(0)):
        tables.append((torch.argmax(logits, dim=1)[i].item(), slice_numbers[k], axis_table[k]))
        log(str(i) + ' ' + str(tables[-1]))
    # Generate activation maps for slices predicted 1
    if predict_masks:
        # def run(command, env={}):
        #     merged_env = os.environ
        #     merged_env.update(env)
        #     process = subprocess.Popen(command, stdout=subprocess.PIPE,
        #                             stderr=subprocess.STDOUT, shell=True,
        #                             env=merged_env)
        #     while True:
        #         line = process.stdout.readline()
        #         line = str(line, 'utf-8')[:-1]
        #         print(line)
        #         if line == '' and process.poll() != None:
        #             break
        #     if process.returncode != 0:
        #         raise Exception("Non zero return code: %d"%process.returncode)
        ##### HELPER FUNCTIONS FOR PLOTTING
        def save_preprocessed_img(fname, preprocessed_imgs, index=0):
            img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
            undo_preprocessed_img = undo_preprocess_input_function(img_copy)
            print('image index {0} in batch'.format(index))
            undo_preprocessed_img = undo_preprocessed_img[0]
            undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
            undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
            
            plt.imsave(fname, undo_preprocessed_img)
            return undo_preprocessed_img

        def save_prototype(fname, epoch, index):
            p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img'+str(index)+'.png'))
            #plt.axis('off')
            plt.imsave(fname, p_img)
            
        def save_prototype_self_activation(fname, epoch, index):
            p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                            'prototype-img-original_with_self_act'+str(index)+'.png'))
            #plt.axis('off')
            plt.imsave(fname, p_img)

        def save_prototype_original_img_with_bbox(fname, epoch, index,
                                                bbox_height_start, bbox_height_end,
                                                bbox_width_start, bbox_width_end, color=(0, 255, 255)):
            p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
            cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                        color, thickness=2)
            p_img_rgb = p_img_bgr[...,::-1]
            p_img_rgb = np.float32(p_img_rgb) / 255
            #plt.imshow(p_img_rgb)
            #plt.axis('off')
            plt.imsave(fname, p_img_rgb)

        def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                            bbox_width_start, bbox_width_end, color=(0, 255, 255)):
            img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
            cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                        color, thickness=2)
            img_rgb_uint8 = img_bgr_uint8[...,::-1]
            img_rgb_float = np.float32(img_rgb_uint8) / 255
            #plt.imshow(img_rgb_float)
            #plt.axis('off')
            plt.imsave(fname, img_rgb_float)
        idx=0
        predicted_cls = tables[idx][0]
        
        ##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
        slice_id = (test_image_path.split("/")[-1]).split(".")[0]
        makedir(os.path.join(save_analysis_path, axis_table[k] + "_" + slice_id, str(slice_numbers[k]) + '_most_activated_prototypes'))
        original_img = save_preprocessed_img(os.path.join(save_analysis_path, axis_table[k] + '_' + str(slice_numbers[k]) + '_original_img.png'),
                                            images_test, idx)
        log('Most activated 10 prototypes of this image:')
        array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
        for i in range(1,11):
            log('top {0} activated prototype for this image:'.format(i))
            save_prototype(os.path.join(save_analysis_path, axis_table[k] + "_" + slice_id, str(slice_numbers[k]) + '_most_activated_prototypes',
                                        'top-%d_activated_prototype.png' % i),
                        start_epoch_number, sorted_indices_act[-i].item())
            save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, axis_table[k] + "_" + slice_id, str(slice_numbers[k]) + '_most_activated_prototypes',
                                                                    'top-%d_activated_prototype_in_original_pimg.png' % i),
                                                epoch=start_epoch_number,
                                                index=sorted_indices_act[-i].item(),
                                                bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
                                                bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
                                                bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
                                                bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
                                                color=(0, 255, 255))
            save_prototype_self_activation(os.path.join(save_analysis_path, axis_table[k] + "_" + slice_id, str(slice_numbers[k]) + '_most_activated_prototypes',
                                                        'top-%d_activated_prototype_self_act.png' % i),
                                        start_epoch_number, sorted_indices_act[-i].item())
            log('prototype index: {0}'.format(sorted_indices_act[-i].item()))
            log('prototype class identity: {0}'.format(prototype_img_identity[sorted_indices_act[-i].item()]))
            if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
                log('prototype connection identity: {0}'.format(prototype_max_connection[sorted_indices_act[-i].item()]))
            log('activation value (similarity score): {0}'.format(array_act[-i]))
            log('last layer connection with predicted class: {0}'.format(ppnet.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))
            
            activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
            upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                    interpolation=cv2.INTER_CUBIC)
            
            # show the most highly activated patch of the image by this prototype
            high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
            high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                        high_act_patch_indices[2]:high_act_patch_indices[3], :]
            log('most highly activated patch of the chosen image by this prototype:')
            plt.axis('off')
            plt.imsave(os.path.join(save_analysis_path, axis_table[k] + "_" + slice_id, str(slice_numbers[k]) + '_most_activated_prototypes',
                                    'most_highly_activated_patch_by_top-%d_prototype.png' % i),
                    high_act_patch)
            log('most highly activated patch by this prototype shown in the original image:')
            imsave_with_bbox(fname=os.path.join(save_analysis_path, axis_table[k] + "_" + slice_id, str(slice_numbers[k]) + '_most_activated_prototypes',
                                    'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                            img_rgb=original_img,
                            bbox_height_start=high_act_patch_indices[0],
                            bbox_height_end=high_act_patch_indices[1],
                            bbox_width_start=high_act_patch_indices[2],
                            bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
            
            # show the image overlayed with prototype activation map
            rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
            rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
            heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[...,::-1]
            overlayed_img = 0.5 * original_img + 0.3 * heatmap
            log('prototype activation map of the chosen image:')
            plt.axis('off')
            plt.imsave(os.path.join(save_analysis_path, axis_table[k] + "_" + slice_id, str(slice_numbers[k]) + '_most_activated_prototypes',
                                    'prototype_activation_map_by_top-%d_prototype.png' % i),
                    overlayed_img)
            log('--------------------------------------------------------------')
        # cmd = "rm %s"%(os.path.join(save_analysis_path, axis_table[k] + '_' + str(slice_numbers[k]) + '_original_img.png'))
        # run(cmd)

if predict_masks == False:
    # get global repartition of preds for this axis
    predictions = [t[0] for t in tables]
    slices_masks = [t[1] for t in tables if t[0] > 0]
    slices_axis = [t[2] for t in tables if t[0] > 0]
    log("Slices predicted 1:" + str(slices_masks))
    log("Corresponding axes: " + str(slices_axis))
    log('Predicted median: ' + str(np.median(predictions)))
    log('Predicted mean: ' + str(np.mean(predictions)))
    label_pred_repartition = [len([el for el in predictions if el==0]), len([el for el in predictions if el==1])]
    label_pred_repartition = [round(label_pred_repartition[0]/(label_pred_repartition[0] + label_pred_repartition[1]), 2), 
                            round(label_pred_repartition[1]/(label_pred_repartition[0] + label_pred_repartition[1]), 2)]
    log("Repartition of predictions: " + str(label_pred_repartition[0]*100) + " percent of slices predicted label 0; " + str(label_pred_repartition[1]*100) + " percent of slices predicted label 1.")    
end = time.time()
log("Processing time: " + str(end - start))

logclose()
torch.cuda.empty_cache()
