import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Conv2D, GlobalAveragePooling2D
from preprocessing import *
from keras.optimizers import Adam,SGD
from keras.applications import Xception, NASNetLarge, NASNetMobile, DenseNet121, DenseNet169, DenseNet201, imagenet_utils

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence, to_categorical
import sys

import numpy as np
from skimage.io import imread
from skimage.transform import resize
import os

from utilities import compute_class_weights

def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf')

# from keras import metrics
# metrics.categorical_accuracy()

class NasNet_Model():
    def __init__(self, input_shape=(299,299,3),  batch_size = 64, num_classes = 100, trainable=True, load_trained=False,
                             is_mobile=False, max_trainable = False, pretrained_model = 'pretrained.h5'):
        try:
            os.mkdir("out_model")
            os.mkdir("logs")
        except:
            print("Created output directory !")
        self.batch_size = batch_size
        self.model = None
        if ~is_mobile:
            self.model = NASNetLarge(input_shape=input_shape, include_top=False,weights='imagenet',pooling='max')
        else:
            self.model = NASNetMobile(input_shape=input_shape, include_top=False,weights='imagenet',pooling='max')

        x = self.model.output
        # x = GlobalAveragePooling2D()(x)
        # add a fully-connected layer
        x = Dense(512, activation='relu')(x)
        self.predictions = Dense(num_classes, activation='softmax', name='out_put')(x)
        # model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        self.model = Model(inputs=self.model.input, outputs=self.predictions)
        
        if load_trained:
            self.model.load_weights(pretrained_model)

        if ~trainable:
            for layer in self.model.layers:
                layer.trainable = False
        if trainable:
            for layer in self.model.layers[:-10]:
                layer.trainable = False
            for layer in self.model.layers[-10:]:
                layer.trainable = True
        
        if max_trainable:
            for layer in self.model.layers:
                layer.trainable = True

        # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=1e-6)
        sgd = SGD(lr=5e-4, decay=1e-6, momentum=0.9, nesterov=True)
        # self.model.compile(optimizer=Adam(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        self.earlyStopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1)
        # self.tensorBoard = keras.callbacks.TensorBoard('./logs',batch_size=batch_size, write_grads=True,write_images=True)
        self.checkpoint = keras.callbacks.ModelCheckpoint('./out_model/weights.{epoch:02d}-{val_acc:.2f}.hdf5',
                                                          monitor='val_acc', verbose=1, save_best_only=True,
                                                          save_weights_only=False, mode='auto', period=1)
        # self.callBackList = [self.earlyStopping, self.tensorBoard, self.checkpoint]
        self.callBackList = [self.earlyStopping, self.checkpoint]

        [self.train_loss, self.train_metrics] = 2*[None]
        self.history = None
        self.dataGenerator = None

    def set_ImageDataGenerator(self, dataGenerator):
        self.dataGenerator = dataGenerator

    def load_model(self, path):
        self.model.load_weights(path)

    def fit(self, x, y, x_val=None, y_val=None):
        if x_val == None:
            self.history = self.model.fit(x, y, batch_size=self.batch_size, epochs=100, verbose=1,
                                          callbacks=self.callBackList, validation_split=0.1)
        else:
            self.history = self.model.fit(x, y, batch_size=self.batch_size, epochs=100, verbose=1,
                                          callbacks=self.callBackList, validation_data=(x_val, y_val))
            
        # self.train_loss, self.train_metrics = self.model.evaluate(x=x, y=y, batch_size=512, verbose=1)
        return self.history

    def fit_generator(self, dataset_dir, target_size = (299, 299), batch_size = 128, nb_of_imgs=90000):
        train_generator = self.dataGenerator.flow_from_directory(dataset_dir, target_size=target_size, batch_size = batch_size ,seed = 110, subset = 'training')
        valid_generator = self.dataGenerator.flow_from_directory(dataset_dir, target_size=target_size, batch_size = batch_size ,seed = 110, subset = 'validation')
        self.model.fit_generator(train_generator, steps_per_epoch = int(nb_of_imgs/batch_size)*1.1, epochs = 70, callbacks=self.callBackList,
                                 validation_data = valid_generator, validation_steps=0.1 * int(nb_of_imgs/batch_size), class_weight=compute_class_weights(dataset_dir))

    def sumary(self):
        return self.model.summary()

    def predict(self, x):
        return self.model.predict(x)

def accuracy(test_x, test_y, model):
    results = model.predict(test_x)
    predicted_labels = np.argmax(results, axis=1)
    true_labels = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_labels == true_labels)
    accuracy = float(num_correct)/results.shape[0]
    return (accuracy * 100)

class MySequence(Sequence):
    def __init__(self, set_file, batch_size, file_size = (299, 299)):
        self.file_size = file_size
        self.batch_size = batch_size
        self.x_set, self.y_set = 2*[None]
        with open(set_file, 'r') as sf:
            self.x_set, self.y_set = sf.readlines.split()
    
    def __len__(self):
        return int(np.ceil(len(self.x_set)/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x_set[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_y = self.y_set[idx*self.batch_size : (idx+1)*self.batch_size]
        return np.array([
            resize(imread(file_name), self.file_size) for file_name in batch_x
        ]), np.array(to_categorical(batch_y))

class MyImageDataGenerator(ImageDataGenerator):
    def __init__(self):
        ImageDataGenerator.__init__(self,height_shift_range=0.2, width_shift_range=0.2, shear_range= 0.2, zoom_range= 0.2,
                                         horizontal_flip=True, validation_split=0.1, preprocessing_function=preprocess_input)


def run():
    dataset_dir = sys.argv[1]
    # dir_test = sys.argv[1]

    # X, Y = load_image(dir_test,num_classes=102, W=299, H=299)

    # model = Xception_Model(input_shape=(299,299,3), 64, 103, trainable=True, pretrained_model = sys.argv[2])
    model = NasNet_Model(input_shape=(331,331,3),  batch_size = 64,
                     num_classes = 103, trainable=True, pretrained_model = sys.argv[2])
    model.sumary()

    dataGenerator = MyImageDataGenerator()
    model.set_ImageDataGenerator(dataGenerator)
    model.fit_generator(dataset_dir, target_size=(331,331), batch_size=64 )

    # model.load_model("weights.05-0.50.hdf5")
    # model.fit(X,Y)


    # print("accuracy on testset: ", accuracy(X,Y,model))
    # print(1)
    # print(2)

if __name__ == '__main__':
    run()
