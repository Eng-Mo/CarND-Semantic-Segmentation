
# coding: utf-8

# In[1]:

# %load main.py
import os
import os.path
import tensorflow as tf
import numpy as np
import helper
import warnings
from distutils.version import LooseVersion
import scipy.misc
import project_tests as tests
import csv
from moviepy.editor import VideoFileClip
from functools import partial

with open('EL.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['Epoch', 'Mean Loss'])


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag],vgg_path)
    
    graph= tf.get_default_graph()
    w1= graph.get_tensor_by_name(vgg_input_tensor_name)
    keep= graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3= graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4= graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7= graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    
    
    
    return w1, keep, layer3, layer4, layer7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    conv_1x1_7= tf.layers.conv2d(vgg_layer7_out,num_classes,1,padding='same',
                                 kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    conv_1x1_4= tf.layers.conv2d(vgg_layer4_out,num_classes,1,padding='same',
                                 kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    conv_1x1_3= tf.layers.conv2d(vgg_layer3_out,num_classes,1,padding='same',
                                 kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    
    output= tf.layers.conv2d_transpose(conv_1x1_7,num_classes,4,(2,2),padding='same',
                                       kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output= tf.add(output,conv_1x1_4)
    
    output= tf.layers.conv2d_transpose(output,num_classes,4,(2,2),padding='same',
                                       kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    outpu= tf.add(output, conv_1x1_3)
    output= tf.layers.conv2d_transpose(output,num_classes,16,(8,8),padding='same',
                                       kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    
    
    
    
    
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer= tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08)
    train_op=optimizer.minimize(cross_entropy_loss)
    
    
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    print("Traning...")
    print()
    count=0
    mean_loss=0
    
    
    for epoch in range(epochs):
       
        for image, label in get_batches_fn(batch_size):
            _, loss= sess.run([train_op,cross_entropy_loss],
                              feed_dict={input_image:image,correct_label:label,keep_prob:.4, learning_rate:.0001})
            print("epoch: {}, loss: {}".format(epoch+1, loss))
            count+=1
            mean_loss+=loss
            
        mean_loss=mean_loss/count
        with open('EL.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1,mean_loss])
        
        mean_loss=0
        count=0
   
            
            
    pass
tests.test_train_nn(train_nn)

def process_image(image, sess, logits, keep_prob, input_image, image_shape):
    
    image = scipy.misc.imresize(image, image_shape)
    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, input_image: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    
    return np.array(street_im)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    print('vgg downloaded')
   

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        epochs=60
        batch_size=5
        input_image, keep_prob, layer3_out, layer4_out, layer7_out= load_vgg(sess, vgg_path)
        layer_out=layers(layer3_out, layer4_out,layer7_out,num_classes)
        label = tf.placeholder(tf.int32, shape=[None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)
        
        
        logits, train_op,loss=optimize(layer_out,label,learning_rate,num_classes)

        # TODO: Train NN using the train_nn function
        
        sess.run(tf.global_variables_initializer())
        train_nn(sess,epochs,batch_size,get_batches_fn,train_op,loss,input_image,label,keep_prob,learning_rate)
        saver = tf.train.Saver()
        saver.save(sess, './runs/sem_seg_model')
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        
        #applying the model to video
        
        video_input1='./data/project_video.mp4'
        video_output1='result1.mp4'
        
        video_input2='./data/challenge_video.mp4'
        video_output2='result2.mp4'
        
        video_input3='./data/harder_challenge_video.mp4'
        video_output3='result3.mp4'
        
        partial_process_image = partial(process_image,  sess=sess, logits=logits, keep_prob=keep_prob,
                                        input_image=input_image, image_shape=image_shape)
        clip1 = VideoFileClip(video_input1)
        video_clip = clip1.fl_image(partial_process_image) #NOTE: this function expects color images!!
        video_clip.write_videofile(video_output1, audio=False)
        
        clip2 = VideoFileClip(video_input2)
        video_clip = clip2.fl_image(partial_process_image) #NOTE: this function expects color images!!
        video_clip.write_videofile(video_output2, audio=False)
        
        clip3 = VideoFileClip(video_input3)
        video_clip = clip3.fl_image(partial_process_image) #NOTE: this function expects color images!!
        video_clip.write_videofile(video_output3, audio=False)
        
        
        
        
     
        


if __name__ == '__main__':
    print("start...")
    run()

