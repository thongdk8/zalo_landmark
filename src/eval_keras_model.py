import tensorflow as tf
import cv2
import os


import time
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(2017) 

import keras
from keras.models import Sequential, load_model
from construct_model import define_model, accuracy
from preprocessing import load_image, map_to_labels

dir_train_folder = r"/home/nghia_VN/TensorFlow_Test/BTL_ML/animal_BTL_ML/train/"
dir_test_folder = r"/home/nghia_VN/TensorFlow_Test/BTL_ML/animal_BTL_ML/test/"
dir_test_folder = r"E:\datasets\animal_BTL_ML\test"
num_classes = 5

#Define Model
model = define_model(input_shape = (32, 32, 3), num_classes = 5)


# Train the model
model.summary()
print ("start loading model")
model = load_model('my_model_v3.h5')
print ("load success")

# features, labels = load_image(dir_train_folder)
# val_index = np.random.choice(len(features), size = 500)
# train_index = np.delete(np.arange(len(features)), val_index)

# val_fea, val_labels = features[val_index], labels[val_index]
# train_features, train_labels = features[train_index], labels[train_index]
test_features, test_labels = load_image(dir_test_folder, train_logical=False)


#loss, metrics = model.evaluate(x=train_features, y=train_labels, batch_size=512, verbose=1)
print("\nEvaluate result ")
#print( loss, metrics)


# test_image = cv2.imread("bird.jpg") 
# test_image = cv2.resize(test_image, (32,32))
# test_image = np.array(test_image)

# test_image2 = cv2.imread("Cat_test.jpg") 
# test_image2 = cv2.resize(test_image2, (32,32))
# test_image2 = np.array(test_image2)
# test_img = np.array([test_image, test_image2])

np.random.seed(10000) #1000
rand_index = np.random.choice(len(test_features), size=12)
test_img, test_img_lables = test_features[rand_index], test_labels[rand_index]
test_img_lables = map_to_labels(test_img_lables)

start = time.time()
#print (map_to_labels(model.predict(np.expand_dims(test_image,0))))
predicted_labels = map_to_labels(model.predict(test_img))
print("Original: ",test_img_lables)
print("Predict:  ",predicted_labels)

end = time.time()


print ("Model took %0.2f seconds to predict"%(end - start))
# compute test accuracy
start = time.time()
print ("\nAccuracy on test data is: %0.2f"%accuracy(test_features, test_labels, model))
end = time.time()
print ("Model took %0.2f seconds to test on %d image"%((end - start), len(test_features)))

print("\nStarting draw testing image and its prediction")
fig = plt.figure()
for i in range(len(test_img)):
    ax = fig.add_subplot(3, 4, 1 + i, xticks=[], yticks=[])
    ax.set_title( "predict: "+predicted_labels[i] +'\n'+"original: "+test_img_lables[i])
    img_to_show = cv2.cvtColor(test_img[i], cv2.COLOR_BGR2RGB)
    plt.imshow(img_to_show)

plt.show()
