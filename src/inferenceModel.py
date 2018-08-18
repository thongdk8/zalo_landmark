import keras
from keras.models import Model, load_model
import utilities as util
import numpy as np

import argparse
import sys
import os

class_mapping_or = {'78': 79, '95': 98, '55': 54, '82': 84, '62': 62, '86': 88, '72': 73,
 '20': 16, '1': 1, '68': 68, '90': 93, '94': 97, '73': 74, '93': 96, '89': 91,
  '102': 5, '97': 100, '33': 30, '25': 21, '5': 48, '91': 94, '84': 86, '42': 40, 
  '51': 50, '70': 71, '83': 85, '30': 27, '34': 31, '18': 13, '0': 0, '7': 70, 
  '99': 102, '61': 61, '31': 28, '59': 58, '76': 77, '24': 20, '79': 80, '32': 29,
   '81': 83, '26': 22, '23': 19, '3': 26, '67': 67, '60': 60, '35': 32, '27': 23, 
   '74': 75, '19': 14, '80': 82, '69': 69, '100': 3, '49': 47, '41': 39, '10': 2, 
   '65': 65, '4': 37, '9': 92, '16': 11, '48': 46, '71': 72, '63': 63, '52': 51, 
   '22': 18, '45': 43, '11': 6, '12': 7, '50': 49, '98': 101, '28': 24, '15': 10, 
   '85': 87, '56': 55, '57': 56, '47': 45, '101': 4, '39': 36, '96': 99, '36': 33, 
   '38': 35, '43': 41, '88': 90, '44': 42, '2': 15, '92': 95, '21': 17, '46': 44, '29': 25,
    '17': 12, '37': 34, '54': 53, '58': 57, '40': 38, '75': 76, '87': 89, '14': 9, 
    '77': 78, '13': 8, '64': 64, '8': 81, '66': 66, '53': 52, '6': 59}

class_mapping = dict(map(reversed, class_mapping_or.items()))

def parse_args(argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, help='path to trained model')
        parser.add_argument('--data', type=str, help='path to test dataset')
        parser.add_argument('--batch_size', type=int, help="size of image batch")
        parser.add_argument('--out', type=str, help='output file name')
        parser.add_argument('--image_size', type=int, help='size of image to feed to network')
        return parser.parse_args(argv)

def run(args):
    myModel = load_model(args.model, compile=False)
    for layer in myModel.layers:
        layer.trainable = False
    myModel.summary()
    
    res_file = open(args.out, 'w')
    res_file.write('id,predicted\n')

    img_files, f_ids = util.list_imgs(args.data)
    nb_sample = len(f_ids)
    print("Number of test image: ", nb_sample)
    nb_steps = int(np.ceil(len(f_ids)/args.batch_size))
    error_ids = []
    for step in range(nb_steps):
        t_imgs, t_ids = [], []
        print("Process with batch ", step)
        for idx in range(args.batch_size):
                
            if((step*args.batch_size + idx) > nb_sample-1):
                break;
            # print("image: ", f_ids[step*args.batch_size + idx])
            try:
                t_img = util.process_image(img_files[step*args.batch_size + idx], img_size=(args.image_size, args.image_size))
                # if (np.asarray(t_img).shape == (299,299,3)):
                t_imgs.append(t_img)
                t_ids.append(f_ids[step*args.batch_size + idx])
            except:
                print("Error: ", f_ids[step*args.batch_size + idx])
                error_ids.append(f_ids[step*args.batch_size + idx])
                continue
                # res_file.write(f_ids[step*args.batch_size + idx]+'\n')
        predicted_res = util.get_top_k_(myModel, np.squeeze(t_imgs))

        for j, idn in enumerate(t_ids):
            res_file.write(str(idn) + ', ' + class_mapping[predicted_res[j][-1]] + ' ' + class_mapping[predicted_res[j][-2]] + ' ' + class_mapping[predicted_res[j][-3]] + '\n')

    for err in error_ids:
        res_file.write(str(err) + ', ' + str(0) + ' ' + str(0) + ' ' + str(0) + '\n')
    res_file.close()
    print("Done....")

if __name__ == '__main__':
    run(parse_args(sys.argv[1:]))