import Transfer_Learning_Xception as baseModel
import utilities as util
import numpy as np

import argparse
import sys

def parse_agrs(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, help='path to trained model')
	parser.add_argument('--data', type=str, help='path to test dataset')
	return parser.parse_args(argv)



def main(args):
        model = baseModel.Xception_Model(input_shape=(299,299,3),  batch_size = 128,
                     num_classes = 103, trainable=False)
        model.sumary()
        model.load_model(args.model)
        imgs = np.squeeze(util.process_image(args.data))
        top1, top2, top3 = util.get_top_k(model, np.expand_dims(imgs, 0))
        print (top1, top2, top3)



if __name__ == '__main__':
	main(parse_agrs(sys.argv[1:]))
    