import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import cv2
from collections import namedtuple
import matplotlib.pyplot as plt




class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num
        
        



def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))
        
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1



Label = namedtuple('Label', ['name', 'color'])

label_defs = [
    Label('unlabeled',     (0,     0,   0)),
    Label('dynamic',       (111,  74,   0)),
    Label('ground',        ( 81,   0,  81)),
    Label('road',          (128,  64, 128)),
    Label('sidewalk',      (244,  35, 232)),
    Label('parking',       (250, 170, 160)),
    Label('rail track',    (230, 150, 140)),
    Label('building',      ( 70,  70,  70)),
    Label('wall',          (102, 102, 156)),
    Label('fence',         (190, 153, 153)),
    Label('guard rail',    (180, 165, 180)),
    Label('bridge',        (150, 100, 100)),
    Label('tunnel',        (150, 120,  90)),
    Label('pole',          (153, 153, 153)),
    Label('traffic light', (250, 170,  30)),
    Label('traffic sign',  (220, 220,   0)),
    Label('vegetation',    (107, 142,  35)),
    Label('terrain',       (152, 251, 152)),
    Label('sky',           ( 70, 130, 180)),
    Label('person',        (220,  20,  60)),
    Label('rider',         (255,   0,   0)),
    Label('car',           (  0,   0, 142)),
    Label('truck',         (  0,   0,  70)),
    Label('bus',           (  0,  60, 100)),
    Label('caravan',       (  0,   0,  90)),
    Label('trailer',       (  0,   0, 110)),
    Label('train',         (  0,  80, 100)),
    Label('motorcycle',    (  0,   0, 230)),
    Label('bicycle',       (119, 11, 32))]




def build_file_list(images_root, labels_root, sample_name):
    image_sample_root = images_root + '/' + sample_name
    image_root_len = len(image_sample_root)
    label_sample_root = labels_root + '/' + sample_name
    image_files = glob(image_sample_root + '/**/*png')
    file_list = []
    for f in image_files:
        f_relative = f[image_root_len:]
        f_dir = os.path.dirname(f_relative)
        f_base = os.path.basename(f_relative)
        f_base_gt = f_base.replace('leftImg8bit', 'gtFine_color')
        f_label = label_sample_root + f_dir + '/' + f_base_gt
        if os.path.exists(f_label):
            file_list.append((f, f_label))
    return file_list


def load_data(data_folder):
    images_root = data_folder + '/leftImg8bit'
    labels_root = data_folder + '/gtFine'

    train_images = build_file_list(images_root, labels_root, 'train')
    valid_images = build_file_list(images_root, labels_root, 'val')
    test_images = build_file_list(images_root, labels_root, 'test')
    num_classes = len(label_defs)
    label_colors = {i: np.array(l.color) for i, l in enumerate(label_defs)}
    image_shape = (256, 512)

    return train_images, valid_images, test_images, num_classes, label_colors, image_shape


def gen_batch_function(image_paths, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        
        
        
       
            
        
        
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            labels = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                
#                 print(image_file)
                
                image = scipy.misc.imresize(scipy.misc.imread(image_file[0]), image_shape)
#                 brigtness_image= augment_brightness_camera_images(image)
                gt_image = scipy.misc.imresize(scipy.misc.imread(image_file[1], mode='RGB'), image_shape)
                
                #fliping image (data Augmentation)
                fliped= np.fliplr(image)
               
                
                #rotation
                #ang_rot = np.random.uniform(20)-20/2
                #rows,cols,ch = image.shape    
                #Rot_image = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
                #image_r = cv2.warpAffine(image,Rot_image,(cols,rows))
                
                                
                #print('image shape= ',image.shape)
                #print('rotated image shape= ',image_r.shape)
                #print('gt image shape= ',gt_image.shape)
                #print('fliped gt image shape = ', flip_gt.shape)
                #

#                 gt_bg = np.all(gt_image == background_color, axis=2)
#                 gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
#                 gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
#                 gt_fliped= np.fliplr(gt_image)
                
                #rows,cols,ch = gt_image.shape    
                #Rot_gt_image = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
                #gt_image_r = cv2.warpAffine(gt_image,Rot_gt_image,(cols,rows))
                
#                 gt_road = np.all(gt_image == road_color, axis=2)
#                 gt_road = gt_road.reshape(*gt_road.shape, 1)
#                 gt_car = np.all(gt_image == car_color, axis=2)
#                 gt_car = gt_car.reshape(*gt_car.shape, 1)
#                 gt_obj = np.concatenate((gt_road, gt_car), axis=2)
#                 gt_bg = np.all(gt_obj == 0, axis=2)
#                 gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
#                 images.append(image)
#                 gt_images.append(gt_image)
                label_bg = np.zeros([image_shape[0], image_shape[1]], dtype=bool)
                label_list = []
                for ldef in label_defs[1:]:

                    label_current = np.all(gt_image == ldef.color, axis=2)
                    label_bg |= label_current
                    label_list.append(label_current)

                label_bg = ~label_bg
                label_all = np.dstack([label_bg, *label_list])
                label_all = label_all.astype(np.float32)

                images.append(image)
                labels.append(label_all)
                
              

#                 images.append(image)
#                 gt_images.append(gt_image)
#                 images.append(fliped)
#                 gt_images.append(gt_fliped)
#                 images.append(brigtness_image)
#                 gt_images.append(gt_image)
                #images.append(image_r)
                #gt_images.append(gt_image_r)
                

            yield np.array(images), np.array(labels)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
        
        
        