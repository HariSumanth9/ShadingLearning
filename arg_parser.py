import argparse

def parse_arguments(*args):
    parser = argparse.ArgumentParser()
	###############added options#######################################
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float,
                        help='Learning rate for the generator')
    
    parser.add_argument('--model', default='encdec', type=str, choices=['texturegan'],
                        help='scribbler|pix2pix')

    parser.add_argument('--num_epoch', default=2000, type=int,
                        help='texture|scribbler')

    parser.add_argument('--no_of_patches', default=2, type=int,
                        help='number of blocks to be removed')

    parser.add_argument('--patch_size', default=[10, 20], type=int,
                        help='size of blocks to be removed')

    parser.add_argument('--patch_multiplier', default=1.5, type=int,
                        help='size multiplier for block to be removed')

    parser.add_argument('--visualize_every', default=10, type=int,
                        help='no. iteration to visualize the results')

    parser.add_argument('--color_space', default='rgb', type=str)
    
    parser.add_argument('--gpu', default=5, type=int, help='GPU ID')

    parser.add_argument('--display_port', default=8057, type=int,
                        help='port for displaying on visdom (need to match with visdom currently open port)')

    parser.add_argument('--data_path', default='./dataset/', type=str,
                        help='path to the data directory, expect train_skg, train_img, val_skg, val_img')

    parser.add_argument('--save_dir', default='./save_dir/', type=str,
                        help='path to save the model')

    parser.add_argument('--load_dir', default='/home/chandu/Pytorch-TextureGAN/save_dir/', type=str,
                        help='path to save the model')

    parser.add_argument('--save_every', default=345, type=int,
                        help='no. iteration to save the models')

    parser.add_argument('--load_epoch', default=-1, type=int,
                        help='The epoch number for the model to load')

    parser.add_argument('--load', default=-1, type=int,
                        help='load generator and discrminator from iteration n')
    
    parser.add_argument('--image_size', default=64, type=int,
                        help='Training images size, after cropping')

    parser.add_argument('--resize_to', default=224, type=int,
                        help='Training images size, after cropping')
                        
    parser.add_argument('--resize_max', default=1, type=float,
                        help='max resize, ratio of the original image, max value is 1')

    parser.add_argument('--resize_min', default=0.6, type=float,
                        help='min resize, ratio of the original image, min value 0')

    parser.add_argument('--patch_size_min', default=40, type=int,
                        help='minumum texture patch size')

    parser.add_argument('--patch_size_max', default=60, type=int,
                        help='max texture patch size')

    parser.add_argument('--batch_size', default=32, type=int,
    					help='Training batch size. MUST BE EVEN NUMBER')

    ############Not Currently Using #################################################################
    parser.add_argument('--tv_weight', default=1, type=float,
                        help='weight ratio for total variation loss')

    parser.add_argument('--mode', default='texture', type=str, choices=['texture', 'scribbler'],
                        help='texture|scribbler')
    
    parser.add_argument('--visualize_mode', default='train', type=str, choices=['train', 'test'],
                        help='train|test')

    parser.add_argument('--crop', default='random', type=str, choices=['random', 'center'],
                        help='random|center')

    parser.add_argument('--contrast', default=True, type=bool,
                        help='randomly adjusting contrast on sketch')

    parser.add_argument('--occlude', default=False, type=bool,
                        help='randomly occlude part of the sketch')

    parser.add_argument('--checkpoints_path', default='data/', type=str,
                        help='output directory for results and models')

    ##################################################################################################################################
    
    return parser.parse_args(*args)
