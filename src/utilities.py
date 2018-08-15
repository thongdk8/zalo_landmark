import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
import cv2
import imageio

def list_imgs(path):
    # images = [os.path.join(path, img) for img in os.listdir(dir) if (img[-4:] == '.png' or img[-4:] == '.jpg')]
    images = []
    img_id = []
    for img in os.listdir(path):
        if (img[-4:] == '.png' or img[-4:] == '.jpg'):
            images.append(os.path.join(path, img))
            img_id.append(img[:-4])
    return images, img_id

def process_image(img_path, img_size = (299,299)):
    img = None
    try:
        img = resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), img_size)
    except:
        print("read gif file ", img_path)
        imgs = imageio.mimread(img_path)
        if (len(np.asarray(imgs[0]).shape) >= 3):
            # print(len(np.asarray(imgs[0]).shape))
            # print(np.asarray(imgs[0]).shape)
            img = cv2.resize(cv2.cvtColor(imgs[0], cv2.COLOR_RGBA2RGB), img_size)
        else:
            img = resize(cv2.cvtColor(imgs[0], cv2.COLOR_GRAY2RGB), img_size)
    img = np.asarray(img, dtype=np.float32)
    # img = img*(1./255)
    return img

def get_top_k(model, imgs, k=3):
    results = model.predict(imgs)
    predicted_labels = np.argsort(results, axis=1)
    top1, top2, top3 = np.squeeze(predicted_labels[:, -k:])
    return top1, top2, top3

def get_top_k_(model, imgs, k=3):
    results = model.predict(imgs)
    # print("Prob", results)
    predicted_labels = np.argsort(results, axis=1)
    # print(predicted_labels)
    return np.squeeze(predicted_labels[:, -k:])

def compute_class_weights(dataset_dir):
    total = 0
    for root, dirs, files in os.walk(dataset_dir):
        total += len(files)

    class_weights = {}
    classes = np.sort([c for c in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, c))] )
    print(classes)
    for i, c in enumerate(classes):
        n_imgs = len( os.listdir(os.path.join(dataset_dir,c)) )
        class_weights[i] = total/(len(classes) * n_imgs)
    
    return class_weights
