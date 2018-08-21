import keras
# from keras import backend as K
from keras import layers
from keras.applications import InceptionResNetV2, Xception
from keras.models import Model
import sys
import os


path_pretrained_model_1 = '/home/thongpb/works/zalo_challenge/models/weights.InceptionResNetV2.52-0.98-0.92.hdf5'
path_pretrained_model_2 = '/home/thongpb/works/zalo_challenge/models/weights.09.acc-0.99-val_acc-0.92.hdf5'

input_shape = (299, 299, 3)
nb_classes = 103
input_img = layers.Input(shape=(input_shape))

x1 = layers.Conv2D(filters=3, kernel_size=(1,1)) (input_img)
x2 = layers.Conv2D(filters=3, kernel_size=(1,1)) (input_img)

model_1 = InceptionResNetV2(input_shape=input_shape, include_top=False,weights=None,pooling='max')
model_2 = Xception(input_shape=input_shape, include_top=False,weights=None,pooling='max')

out_model_1 = model_1.output
dense_1 = layers.Dense(1024, activation='relu') (out_model_1)
out_1 = layers.Dense(nb_classes, activation='softmax',name='out_put') (dense_1)

out_model_2 = model_2.output
dense_2 = layers.Dense(1024, activation='relu') (out_model_2)
out_2 = layers.Dense(nb_classes, activation='softmax',name='out_put2') (dense_2)

M1 = Model(inputs=model_1.input, outputs=out_1)
# M1.load_weights(path_pretrained_model_1)
M1 = Model(inputs=M1.input, outputs=dense_1)
# M1.layers.pop()
# M1.build(input_shape)
re_m1 = M1(input_img)


M2 = Model(inputs=model_2.input, outputs=out_2)
# M2.load_weights(path_pretrained_model_2)
M2 = Model(inputs=M2.input, outputs=dense_2)
# M2.layers.pop()
# M2.build(input_shape)
re_m2 = M2(input_img)

M1 = Model(inputs=input_img, outputs=re_m1)
M2 = Model(inputs=input_img, outputs=re_m2)
M1.summary()
M2.summary()

x = layers.concatenate( [M1.output, M2.output] )

x2 = layers.Dense(1024, activation='relu') (x)

predictions = layers.Dense(nb_classes,activation='softmax') (x2)

print('set input')

# seq_M = keras.Sequential()

# for layer in M1.layers:
#     seq_M.add(layer)
# for layer in M2.layers:
#     seq_M.add(layer)

# seq_M.add(x)
# seq_M.add(x2)
# seq_M.add(predictions)

fn_model = Model(inputs=input_img, outputs=predictions)

fn_model.summary()




# from keras.utils import plot_model
# plot_model(fn_model, to_file='model.png')
# for layer in fn_model.layers:
#     print(layer.name)