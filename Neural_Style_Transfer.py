# Dhruvin Soni


import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
from keras.preprocessing import image


#from google.colab import files
#uploaded = files.upload()

target_image_path = 'portrait.jpg'
style_reference_image_path = 'transfer_style_reference.jpg'

width, height = load_img(target_image_path).size
print(width, height)

img_height = 400
img_width = int(width * img_height / height)
print(img_height, img_width)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    # zero-centering by removing the mean pixel value from ImageNet
    # this reverses a transformation done by vgg19.preprocess_input
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.680
    # Convergs image from 'BGR' to 'RGB', this is also part of vgg19.preprocess_input reversal process
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

from keras import backend as K

target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))
# placeholder that will contain the generated image
combination_image = K.placeholder((1, img_height, img_width, 3))

# combines the three images in a single batch
input_tensor = K.concatenate([target_image,
                              style_reference_image,
                              combination_image], axis=0)

# builds the VGG19 network with the batch of three images as input.
# the model will be loaded with pretrained imagenet weights
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet',
                    include_top=False)
print('Model loaded.')

def content_loss(base, combination):
    return K.sum(K.square(combination - base))

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))
  
def total_variation_loss(x):
    a = K.square(
          x[:, :img_height - 1, :img_width - 1, :] -
          x[:, 1:, :img_width - 1, :])
    b = K.square(
          x[:, :img_height - 1, :img_width - 1, :] -
          x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# dictionary that maps layer names to activation tensors
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# layer used for content losst
content_layer = 'block5_conv2'

# layers used for style loss
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

# weights in the weighted average
total_variation_weight = 1e-4
style_weight = 1.0
content_weight = 0.025

# define loss by adding all components to this scalar variable
loss = K.variable(0.0)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss = loss + (content_weight * content_loss(target_image_features,
                                     combination_features))

# adds a style loss component for each target layer
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl
    
# adds the total variation loss
loss = loss + (total_variation_weight * total_variation_loss(combination_image))

# gets the gradients of the generated image with regard
# to the loss
grads = K.gradients(loss, combination_image)[0]

# function to fetch the values of the current loss and the
# current gadients
fetch_loss_and_grads = K.function([combination_image], [loss, grads])

# this class wraps fetch_loss_and_grads
# in a way that lets you retrieve the losses and
# gradients via two separate mthod calls, which is
# required by the SciPy optimizer you'll use
class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grad_values = None
        
    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
    
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

result_prefix = 'my_result'
iterations = 25

# this is the initial state: the target image
x = preprocess_image(target_image_path)
# you flatten image because L-BFGS can only process flat vectors
x = x.flatten()

# run L-BFGS optimization ove rthe pixels of the
# generated image to minimize the neural style
# loss.  NOTE: we pass the function to evaluate
# the loss and gradients as two separate args
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss,
                                     x,
                                     fprime=evaluator.grads,
                                     maxfun=20)
    print('Current loss value:', min_val)
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    print('Image saved as', fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))

    
