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

type_models = {0:'DenseNet121', 1:'DenseNet169', 2:'DenseNet201', 3:'ResNet50', 4:'InceptionV3', 5:'InceptionResNetV2', 6:'NASNetLarge', 7: 'NASNetMobile'}

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

        if net_type==0:
            self.model = DenseNet121(input_shape=input_shape, include_top=False,weights='imagenet',pooling='max')
            self.pre_process = keras.applications.densenet.preprocess_input
        elif net_type==1:
            self.model = DenseNet169(input_shape=input_shape, include_top=False,weights='imagenet',pooling='max')
            self.pre_process = keras.applications.densenet.preprocess_input
        elif net_type==2:
            self.model = DenseNet201(input_shape=input_shape, include_top=False,weights='imagenet',pooling='max')
            self.pre_process = keras.applications.densenet.preprocess_input
        elif net_type==3:
            self.model = ResNet50(input_shape=input_shape, include_top=False,weights='imagenet',pooling='max')
            self.pre_process = keras.applications.resnet50.preprocess_input
        elif net_type==4:
            self.model = InceptionV3(input_shape=input_shape, include_top=False,weights='imagenet',pooling='max')
            self.pre_process = keras.applications.inception_v3.preprocess_input
        elif net_type==5:
            self.model = InceptionResNetV2(input_shape=input_shape, include_top=False,weights='imagenet',pooling='max')
            self.pre_process = keras.applications.inception_resnet_v2.preprocess_input
        elif net_type==6:
            self.model = NASNetLarge(input_shape=input_shape, include_top=False,weights='imagenet',pooling='max')
            self.pre_process = keras.applications.nasnet.preprocess_input
        elif net_type==7:
            self.model = NASNetMobile(input_shape=input_shape, include_top=False,weights='imagenet',pooling='max')
            self.pre_process = keras.applications.nasnet.preprocess_input

        x = self.model.output
        # x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x) # add a fully-connected layer
        self.predictions = Dense(num_classes, activation='softmax', name='out_put')(x)
        self.model = Model(inputs=self.model.input, outputs=self.predictions)
        
        if load_trained:
            self.model.load_weights(pretrained_model)
            print("Load pretrained model successfully!")

        if trainable == False:
            for layer in self.model.layers:
                layer.trainable = False
            print("Use model for inference is activated!")
        if trainable and not max_trainable:
            for layer in self.model.layers[:-5]:
                layer.trainable = False
            for layer in self.model.layers[-5:]:
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

        self.earlyStopping = callbacks.EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1)
        self.tensorBoard = callbacks.TensorBoard('./logs',batch_size=batch_size, write_grads=True,write_images=True)
        self.checkpoint = callbacks.ModelCheckpoint('./out_model/weights.' + type_models[self.net_type] + '.{epoch:02d}-{acc:.2f}-{val_acc:.2f}.hdf5',
                                                          monitor='val_acc', verbose=1, save_best_only=True,
                                                          save_weights_only=False, mode='auto', period=1)
        self.lrController = callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, verbose=1, mode='auto', 
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

        train_generator = self.dataGenerator.flow_from_directory(dataset_dir, target_size=target_size, batch_size = batch_size ,seed = 110, subset = 'training')
        valid_generator = self.dataGenerator.flow_from_directory(dataset_dir, target_size=target_size, batch_size = batch_size ,seed = 110, subset = 'validation')
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

    dataGenerator = MyImageDataGenerator(model.get_pre_process_func())
    model.set_ImageDataGenerator(dataGenerator)
    model.fit_generator_from_directory(dataset_dir)
    print("trainning is done!")
    model.write_history()

if __name__ == '__main__':
    run(parse_args(sys.argv[1:]))
