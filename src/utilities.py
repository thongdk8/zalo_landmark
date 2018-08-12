import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize

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
    img = resize(imread(img_path), img_size)
    img = np.asarray(img, dtype=np.float32)
    img = img*(1./255)
    return [img]

def get_top_k(model, imgs, k=3):
    results = model.predict(imgs)
    predicted_labels = np.argsort(results, axis=1)
    top1, top2, top3 = np.squeeze(predicted_labels[:, -k:])
    return top1, top2, top3