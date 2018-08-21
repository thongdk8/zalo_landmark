import keras
# from keras import backend as K
from keras import layers
from keras.applications import InceptionResNetV2, Xception
from keras.models import Model
import sys
import os

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Conv2D, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from keras.applications import Xception, NASNetLarge, NASNetMobile, DenseNet121, DenseNet169, DenseNet201
from keras.applications import InceptionV3, InceptionResNetV2, imagenet_utils, ResNet50
from keras import callbacks

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence, to_categorical

import numpy as np
from skimage.io import imread
from skimage.transform import resize

from utilities import compute_class_weights, count_image_samples

import argparse
import sys
import os

path_pretrained_model_1 = '/home/pbthong/works/zalo_challenge/zalo_landmark/src/out_model/weights.InceptionResNetV2.52-0.98-0.92.hdf5'
path_pretrained_model_2 = '/home/pbthong/works/zalo_challenge/weights.31.acc-0.98-val_acc-0.93.hdf5'


K.set_floatx('float32')


type_models = {0:'Custom', 1:'DenseNet169', 2:'DenseNet201', 3:'ResNet50', 4:'InceptionV3', 5:'InceptionResNetV2', 6:'NASNetLarge', 7: 'NASNetMobile', 8:'Xception'}

def parse_args(argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, help='path to trained model if load_trained is specified')
        parser.add_argument('--data', type=str, help='path to dataset')
        parser.add_argument('--batch_size', type=int, default=64, help="size of image batch")
        parser.add_argument('--init_lr', type=float, default=0.001,help="init learning rate")
        parser.add_argument('--max_trainable', type=lambda x: (str(x).lower() == 'true'), default=False, help="is trainable whole network")
        parser.add_argument('--trainable', type=lambda x: (str(x).lower() == 'true'), default=True, help="is training or inference")
        parser.add_argument('--load_trained', type=lambda x: (str(x).lower() == 'true'), default=False, help="is load trained model for tuning")
        parser.add_argument('--image_size', type=int, default=224, help="size of input image feed to model")
        parser.add_argument('--optimizer', type=str, default='adam', help='optimizer use for training')
        parser.add_argument('--init_epoch', type=int, default=0, help='init epoch if run from previous running')
        parser.add_argument('--max_epoch', type=int, default=100, help='max of epoch')
        parser.add_argument('--net_type', type=int, default=0, 
            help='id of network: 0:DenseNet121, 1:DenseNet169, 2:DenseNet201, 3:ResNet50, 4:InceptionV3, 5:InceptionResNetV2, 6:NASNetLarge, 7: NASNetMobile')
        parser.add_argument('--nb_classes', type=int, default=103, help='number of class')
        return parser.parse_args(argv)


class MyModel():
    def __init__(self, image_size=299,  batch_size = 64, num_classes = 100, trainable=True, load_trained=False,
                              max_trainable = False, pretrained_model = 'pretrained.h5',
                              init_lr = 0.001, n_chanels=3, optimizer='adam', init_epoch=0, max_epoch=100, net_type=0):
        try:
            os.mkdir("out_model")
            os.mkdir("logs")
        except:
            print("Created output directory !")
        
        self.image_size = image_size
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.max_epoch = max_epoch
        self.init_epoch = init_epoch
        self.net_type = net_type

        self.model = None
        self.pre_process = None
        
        input_shape = (image_size, image_size, n_chanels)


        nb_classes = num_classes
        input_img = layers.Input(shape=(input_shape))

        model_1 = InceptionResNetV2(input_shape=input_shape, include_top=False,weights=None,pooling='max')
        model_2 = Xception(input_shape=input_shape, include_top=False,weights=None,pooling='max')

        out_model_1 = model_1.output
        dense_1 = layers.Dense(1024, activation='relu') (out_model_1)
        out_1 = layers.Dense(nb_classes, activation='softmax',name='out_put') (dense_1)

        out_model_2 = model_2.output
        dense_2 = layers.Dense(1024, activation='relu') (out_model_2)
        out_2 = layers.Dense(nb_classes, activation='softmax',name='out_put2') (dense_2)

        M1 = Model(inputs=model_1.input, outputs=out_1)
        M1.load_weights(path_pretrained_model_1)
        M1 = Model(inputs=M1.input, outputs=dense_1)
        # M1.layers.pop()
        # M1.build(input_shape)
        # K.reset_uids()
        re_m1 = M1(input_img)


        M2 = Model(inputs=model_2.input, outputs=out_2)
        M2.load_weights(path_pretrained_model_2)
        M2 = Model(inputs=M2.input, outputs=dense_2)
        # M2.layers.pop()
        # M2.build(input_shape)
        # K.reset_uids()
        re_m2 = M2(input_img)

        M1 = Model(inputs=input_img, outputs=re_m1)
        M2 = Model(inputs=input_img, outputs=re_m2)

        x = layers.concatenate( [M1.output, M2.output] )
        x = layers.BatchNormalization()
        x2 = layers.Dense(2048, activation='relu') (x)
        x2 = layers.BatchNormalization()
        predictions = layers.Dense(nb_classes,activation='softmax') (x2)

        # K.reset_uids()
        self.model = Model(inputs=input_img, outputs=predictions)
        
        if load_trained:
            self.model.load_weights(pretrained_model)
            print("Load pretrained model successfully!")

        if trainable == False:
            for layer in self.model.layers:
                layer.trainable = False
            print("Use model for inference is activated!")
        if trainable and not max_trainable:
            for layer in self.model.layers[:-3]:
                try:
                    layer.trainable = False
                except:
                    try:
                        print("Freezing submodel")
                        for l in layer.layers:
                            l.trainable = False
                    except:
                        print ("cannot freeze layer")
                        pass
            for layer in self.model.layers[-3:]:
                layer.trainable = True
            print("Train last layers is activated!")
        if max_trainable:
            for layer in self.model.layers:
                layer.trainable = True
            print("Train whole network is activated!")
        
        if (optimizer=='adam'):
            opt = Adam(lr=init_lr, beta_1=0.9, beta_2=0.999, decay=1e-6)
        else:
            opt = SGD(lr=init_lr, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        self.earlyStopping = callbacks.EarlyStopping(monitor='val_acc', min_delta=0.001, patience=15, verbose=1)
        self.tensorBoard = callbacks.TensorBoard('./logs',batch_size=batch_size, write_grads=True,write_images=True)
        self.checkpoint = callbacks.ModelCheckpoint('./out_model/weights.' + type_models[self.net_type] + '.{epoch:02d}-{acc:.2f}-{val_acc:.2f}.hdf5',
                                                          monitor='val_acc', verbose=1, save_best_only=True,
                                                          save_weights_only=False, mode='auto', period=1)
        self.lrController = callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.7, patience=3, verbose=1, mode='auto', 
                                                                min_delta=0.0001, cooldown=0, min_lr=0.00001)
        self.history_ = callbacks.History()
        self.callBackList = [self.earlyStopping, self.tensorBoard, self.checkpoint, self.lrController, self.history_]

        # [self.train_loss, self.train_metrics] = 2*[None]
        self.history = None
        self.dataGenerator = None

    def set_ImageDataGenerator(self, dataGenerator):
        self.dataGenerator = dataGenerator

    def load_model(self, path):
        self.model.load_weights(path, by_name=False)

    def fit_generator_from_directory(self, dataset_dir):
        batch_size = self.batch_size
        target_size = (self.image_size, self.image_size)
        nb_of_imgs = count_image_samples(dataset_dir)
        print("Start training with learning rate=", self.init_lr, 'batch size=', batch_size, 'on {} image samples.'.format(nb_of_imgs))

        train_generator = self.dataGenerator.flow_from_directory(dataset_dir, target_size=target_size, batch_size = batch_size ,seed = 120, subset = 'training')
        valid_generator = self.dataGenerator.flow_from_directory(dataset_dir, target_size=target_size, batch_size = batch_size ,seed = 120, subset = 'validation')
        print("Class mapping: ")
        print(train_generator.class_indices)
        self.model.fit_generator(train_generator, steps_per_epoch = int(nb_of_imgs/batch_size)*1.1, epochs = self.max_epoch, 
                                  callbacks=self.callBackList,
                                 validation_data = valid_generator, validation_steps=0.1 * int(nb_of_imgs/batch_size), initial_epoch = self.init_epoch,
                                 class_weight=compute_class_weights(dataset_dir))

    def sumary(self):
        return self.model.summary()

    def predict(self, x):
        return self.model.predict(x)

    def get_pre_process_func(self):
        return self.pre_process

    def write_history(self):
        with open('./logs/' + type_models[self.net_type] + '-batch_size-' + str(self.batch_size), 'w') as hf:
            hf.write(str(self.history_.history))


class MyImageDataGenerator(ImageDataGenerator):
    def __init__(self, pre_process):
        ImageDataGenerator.__init__(self, height_shift_range=0.2, width_shift_range=0.2, shear_range= 0.2, zoom_range= 0.2, 
                                    horizontal_flip=True, validation_split=0.1, rotation_range = 30, 
                                    preprocessing_function=pre_process)

def run(args):
    dataset_dir = args.data

    model = MyModel(image_size=args.image_size,  batch_size = args.batch_size,
                     num_classes = args.nb_classes, trainable=args.trainable, pretrained_model = args.model,
                     init_lr=args.init_lr, max_trainable=args.max_trainable, load_trained=args.load_trained, 
                     init_epoch=args.init_epoch, optimizer=args.optimizer, max_epoch=args.max_epoch, net_type=args.net_type)
    model.sumary()

    dataGenerator = MyImageDataGenerator(keras.applications.inception_resnet_v2.preprocess_input)
    model.set_ImageDataGenerator(dataGenerator)
    model.fit_generator_from_directory(dataset_dir)
    print("trainning is done!")
    model.write_history()

if __name__ == '__main__':
    run(parse_args(sys.argv[1:]))
