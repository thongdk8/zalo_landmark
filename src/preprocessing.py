import cv2
import os
import numpy as np

def to_arr_target(labels, num_classes = 5):
    temp = np.zeros(num_classes)
    temp[labels-1] = 1
    return temp

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def preprocess_image(path, W, H):
    try:
        print ("Read image: " , path)
        readImage = cv2.imread(path)
        #readImage = cv2.cvtColor(readImage, cv2.COLOR_BGR2GRAY)
        readImage = cv2.resize(readImage,(W,H))
        image_arr = np.asarray(readImage)
        image_arr = np.array(image_arr, dtype=np.float32)
        image_arr = preprocess_input(image_arr)
        element_folder = str(path).split('/')
        # element_folder = str(path).split('\\')
        label = int(element_folder[-2])

        return image_arr, label
    except:
        return None, None



def load_image(dir_folder,num_classes = 5, train_logical=True, W=32, H=32):
    if train_logical:
        files = []
        for i in range(0, num_classes+1):
            fileDir = os.path.join(dir_folder,'{}'.format(str(i)))
            for name in os.listdir(fileDir):
                if os.path.isfile(os.path.join(fileDir, name)):
                    files.append(os.path.join(fileDir, name))
    else:
        files = []
        for i in range(1, num_classes+1):
            fileDir = os.path.join(dir_folder,'{}'.format(str(i)))
            for name in os.listdir(fileDir):
                if os.path.isfile(os.path.join(fileDir, name)):
                    files.append(os.path.join(fileDir, name))

    images = []
    labels = []
    number_of_image = len(files)
    for i in range(0,number_of_image):
        t_image, t_label = preprocess_image(files[i], W, H)
        if t_image != None:
            images.append(t_image)
            labels.append(t_label)
    images = np.array([x for x in images])
    labels = np.array([to_arr_target(x) for x in labels])
    print('Shape images is ',np.shape(images))
    print('Shape labels is ',np.shape(labels))
    return (images, labels)

def map_to_labels(logits, dir_labels=r"E:\datasets\animal_BTL_ML\labels.txt"):
    labels = []
    # with open("./animal_BTL_ML/labels.txt") as lb_file:
    with open(dir_labels) as lb_file:
        lines = lb_file.read().splitlines()

    id_labels = np.argmax(logits, axis=1)
    for i in range(len(id_labels)):
        labels.append(str(lines[id_labels[i]]))
    return labels