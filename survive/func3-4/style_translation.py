import argparse
from utils import *
import os
from tqdm import tqdm
from glob import glob
import time
import numpy as np
import generator
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    desc = "Tensorflow implementation of AnimeGAN"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--checkpoint_dir', type=str, default='/root/CSC4001/checkpoint/style',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--test_dir', type=str, default='/root/CSC4001/data/test_img',
                        help='Directory name of test photos')
    parser.add_argument('--style_name', type=str, default='style',
                        help='what style you want to get')

    """checking arguments"""

    return parser.parse_args()

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    # params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {}'.format(flops.total_float_ops))

def image_style_translation(checkpoint_dir='/root/CSC4001/checkpoint/style', style_name='style', test_dir='', img_size=[256,256]):
    # tf.reset_default_graph()
    result_dir = 'results/'+style_name
    check_folder(result_dir)
    # test_files = glob('{}/*.*'.format(test_dir))
    test_files = [test_dir]

    # test_real = tf.placeholder(tf.float32, [1, 256, 256, 3], name='test')
    test_real = tf.placeholder(tf.float32, [1, None, None, 3], name='test')

    with tf.variable_scope("generator", reuse=False):
        test_generated = generator.G_net(test_real).fake
    saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        # tf.global_variables_initializer().run()
        # load model
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # checkpoint file information
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # first line
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
        else:
            print(checkpoint_dir)
            print(ckpt)
            print(" [*] Failed to find a checkpoint")
            return
        
        # FLOPs
        stats_graph(tf.get_default_graph())

        begin = time.time()
        for sample_file  in tqdm(test_files) :
            # print('Processing image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, img_size))
            # image_path = os.path.join(result_dir,'{0}'.format(os.path.basename(sample_file)))
            image_path = os.path.join(result_dir,'{0}'.format(os.path.basename(sample_file).split('.')[0]+'.jpg'))
            fake_img = sess.run(test_generated, feed_dict = {test_real : sample_image})
            save_images(fake_img, image_path)
        end = time.time()
        print(f'test-time: {end-begin} s')
        print(f'one image test time : {(end-begin)/len(test_files)} s')
        return image_path

if __name__ == '__main__':
    arg = parse_args()
    print(arg.checkpoint_dir)
    image_style_translation(arg.checkpoint_dir, arg.style_name, arg.test_dir)
