import keras.backend as K
from keras.layers import Input, Flatten, Reshape, concatenate
from keras.layers import Dense, Conv2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, Activation
from keras.models import Model

from ssd_util import Normalize, PriorBox

def SSD(input_shape, num_classes):

    model = {}

    model['in'] = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])
    
    model['conv1.1'] = Conv2D(64, 3, 3,
            activation='relu')(model['in'])
    model['conv1.2'] = Conv2D(64, 3, 3,
            activation='relu')(model['conv1.1'])
    model['pool1'] = MaxPooling2D(pool_size=(2,2),
                    strides=(2,2))(model['conv1.2'])
    
    model['conv2.1'] = Conv2D(128, 3, 3,
            activation='relu')(model['pool1'])
    model['conv2.2'] = Conv2D(128, 3, 3,
            activation='relu')(model['conv2.1'])
    model['pool2'] = MaxPooling2D(pool_size=(2,2),
                    strides=(2,2))(model['conv2.2'])

    model['conv3.1'] = Conv2D(256, 3, 3,
            activation='relu')(model['pool2'])
    model['conv3.2'] = Conv2D(256, 3, 3,
            activation='relu')(model['conv3.1'])
    model['pool3'] = MaxPooling2D(pool_size=(2,2),
                    strides=(2,2))(model['conv3.2'])
    
    model['conv4.1'] = Conv2D(512, 3, 3,
            activation='relu')(model['pool3'])
    model['conv4.2'] = Conv2D(512, 3, 3,
            activation='relu')(model['conv4.1'])
    model['pool4'] = MaxPooling2D(pool_size=(2,2),
                    strides=(2,2))(model['conv4.2'])
    
    model['conv5.1'] = Conv2D(512, 3, 3,
            activation='relu')(model['pool4'])
    model['conv5.2'] = Conv2D(512, 3, 3,
            activation='relu')(model['conv5.1'])
    #model['pool5'] = MaxPooling2D(pool_size=(2,2),
    #                strides=(2,2))(model['conv5.2'])
    
    model['fc1'] = Conv2D(1024, 3, 3,
            activation='relu')(model['conv5.2'])
    model['fc2'] = Conv2D(1024, 1, 1,
            activation='relu')(model['fc1'])

    model['conv6.1'] = Conv2D(256, 1, 1,
            activation='relu')(model['fc2'])
    model['conv6.2'] = Conv2D(512, 3, 3,
            activation='relu')(model['conv6.1'])
    
    model['conv7.1'] = Conv2D(128, 1, 1,
            activation='relu')(model['conv6.2'])
    model['conv7.2'] = Conv2D(256, 3, 3,
            activation='relu')(model['conv7.1'])
    
    model['conv8.1'] = Conv2D(128, 1, 1,
            activation='relu')(model['conv7.2'])
    model['conv8.2'] = Conv2D(256, 3, 3,
            activation='relu')(model['conv8.1'])

    # Last
    model['gap'] = GlobalAveragePooling2D()(model['conv8.2'])

    # prediction from conv4
    model['conv4.2_norm'] = Normalize(20)(model['conv4.2'])

    num_priors = 3
    x = Conv2D(num_priors * 4, 3, 3)(model['conv4.2_norm'])
    model['conv4.2_norm_loc_flat'] = Flatten()(x)

    x = Conv2D(num_priors * num_classes, 3, 3)(model['conv4.2_norm'])
    model['conv4.2_norm_conf_flat'] = Flatten()(x)

    model['conv4.2_priorbox'] = PriorBox(img_size, 30.0,
                        aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2])(model['conv4.2_norm'])
    
    # prediction from fc2
    num_priors = 6
    model['fc2_loc'] = Conv2D(num_priors*4, 3, 3)(model['fc2'])
    model['fc2_loc_flat'] = Flatten()(model['fc2_loc'])

    model['fc2_conf'] = Conv2D(num_priors*num_classes, 3, 3)(model['fc2'])
    model['fc2_conf_flat'] = Flatten()(model['fc2_conf'])

    model['fc2_priorbox'] = PriorBox(img_size, 60.0, max_size=114.0,
                            aspect_ratios=[2, 3],
                            variances=[0.1, 0.1, 0.2, 0.2])(model['fc2'])
    
    # precition from conv6.2
    num_priors = 6
    model['conv6.2_loc'] = Conv2D(num_priors*4, 3, 3)(model['conv6.2'])
    model['conv6.2_loc_flat'] = Flatten()(model['conv6.2_loc'])

    model['conv6.2_conf'] = Conv2D(num_priors*num_classes, 3, 3)(model['conv6.2'])
    model['conv6.2_conf_flat'] = Flatten()(model['conv6.2_conf'])

    model['conv6.2_priorbox'] = PriorBox(img_size, 114.0, max_size=168.0,
                            aspect_ratios=[2, 3],
                            variances=[0.1, 0.1, 0.2, 0.2])(model['conv6.2'])
    
    # prediction from conv7.2
    num_priors = 6
    model['conv7.2_loc'] = Conv2D(num_priors*4, 3, 3)(model['conv7.2'])
    model['conv7.2_loc_flat'] = Flatten()(model['conv7.2_loc'])

    model['conv7.2_conf'] = Conv2D(num_priors*num_classes, 3, 3)(model['conv7.2'])
    model['conv7.2_conf_flat'] = Flatten()(model['conv7.2_conf'])

    model['conv7.2_priorbox'] = PriorBox(img_size, 168.0, max_size=222.0,
                            aspect_ratios=[2, 3],
                            variances=[0.1, 0.1, 0.2, 0.2])(model['conv7.2'])
    
    # prediction from conv8.2
    num_priors = 6
    model['conv8.2_loc'] = Conv2D(num_priors*4, 3, 3)(model['conv8.2'])
    model['conv8.2_loc_flat'] = Flatten()(model['conv8.2_loc'])

    model['conv8.2_conf'] = Conv2D(num_priors*num_classes, 3, 3)(model['conv8.2'])
    model['conv8.2_conf_flat'] = Flatten()(model['conv8.2_conf'])

    model['conv8.2_priorbox'] = PriorBox(img_size, 222.0, max_size=276.0,
                            aspect_ratios=[2, 3],
                            variances=[0.1, 0.1, 0.2, 0.2])(model['conv8.2'])
    
    # prediction from gap
    num_priors = 6
    model['gap_loc'] = Dense(num_priors*4)(model['gap'])
    model['gap_conf'] = Dense(num_priors * num_classes)(model['gap'])

    if K.image_dim_ordering() == 'tf':
        target_shape = (1,1,256)
    else:
        target_shape = (256, 1, 1)
    
    model['gap_priorbox'] = PriorBox(img_size, 276.0, max_size=330.0,
                            aspect_ratios=[2, 3],
                            variances=[0.1, 0.1, 0.2, 0.2]
                            )(Reshape(target_shape)(model['gap']))
    
    
    # gather all predictions
    model['loc'] = concatenate([
        model['conv4.2_norm_loc_flat'],
        model['fc2_loc_flat'],
        model['conv6.2_loc_flat'],
        model['conv7.2_loc_flat'],
        model['conv8.2_loc_flat'],
        model['gap_loc']
    ], axis=1)

    model['conf'] = concatenate([
        model['conv4.2_norm_conf_flat'],
        model['fc2_conf_flat'],
        model['conv6.2_conf_flat'],
        model['conv7.2_conf_flat'],
        model['conv8.2_conf_flat'],
        model['gap_conf']
    ], axis=1)

    model['priorbox'] = concatenate([
        model['conv4.2_priorbox'],
        model['fc2_priorbox'],
        model['conv6.2_priorbox'],
        model['conv7.2_priorbox'],
        model['conv8.2_priorbox'],
        model['gap_priorbox']
    ], axis=1)

    if hasattr(model['loc'], '_keras_shape'):
        num_boxes = model['loc']._keras_shape[-1] // 4
    elif hasattr(model['loc'], 'int_shape'):
        num_boxes = K.int_shape(model['loc'])[-1] // 4
    
    model['loc'] = Reshape((num_boxes,4))(model['loc'])

    model['conf'] = Reshape((num_boxes,num_classes))(model['conf'])
    model['conf'] = Activation('softmax')(model['conf'])
    """
    model['predictions'] = concatenate([
        model['loc'],
        model['conf'],
        model['priorbox']
    ], axis=2)
    """

    return Model(model['in'], [model['loc'],model['conf'], model['priorbox']])