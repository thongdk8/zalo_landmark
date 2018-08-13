import Transfer_Learning_Xception as baseModel
import utilities as util
import numpy as np

import argparse
import sys

def parse_agrs(argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, help='path to trained model')
        parser.add_argument('--data', type=str, help='path to test dataset')
        parser.add_argument('--batch_size', type=int, help="size of image batch")
        parser.add_argument('--out', type=str, help='output file name')
        return parser.parse_args(argv)



def main(args):
        model = baseModel.Xception_Model(input_shape=(299,299,3),  batch_size = 128,
                        num_classes = 103, trainable=False)
        model.sumary()
        model.load_model(args.model)

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
                                t_img = util.process_image(img_files[step*args.batch_size + idx])
                                # if (np.asarray(t_img).shape == (299,299,3)):
                                t_imgs.append(t_img)
                                t_ids.append(f_ids[step*args.batch_size + idx])
                        except:
                                print("Error: ", f_ids[step*args.batch_size + idx])
                                error_ids.append(f_ids[step*args.batch_size + idx])
                                continue
                                # res_file.write(f_ids[step*args.batch_size + idx]+'\n')
                predicted_res = util.get_top_k_(model, np.squeeze(t_imgs))

                for j, idn in enumerate(t_ids):
                        res_file.write(str(idn) + ', ' + str(predicted_res[j][-1]) + ' ' + str(predicted_res[j][-2]) + ' ' + str(predicted_res[j][-3]) + '\n')

        for err in error_ids:
                res_file.write(str(err) + ', ' + str(0) + ' ' + str(0) + ' ' + str(0) + '\n')
        res_file.close()
        print("Done....")
        # imgs = np.squeeze(util.process_image(args.data))
        # top1, top2, top3 = util.get_top_k(model, np.expand_dims(imgs, 0))
        # print (top1, top2, top3)



if __name__ == '__main__':
	main(parse_agrs(sys.argv[1:]))
    