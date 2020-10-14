import os
import json
import math

import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm

import csv

from dataset import Dataset
from models import GeneratorDCGAN, GeneratorResNet
from utils import set_chars_type, concat_imgs, make_gif, Translater

FLAGS = tf.app.flags.FLAGS

class GeneratingFontDesignGAN():
    """Generating font design GAN

    This class is only for generating fonts.
    """

    def __init__(self):
        global FLAGS
        self._setup_dirs()
        self._setup_params()
        self._setup_embedding_chars()
        self.translater = Translater(num_class=52)
        if FLAGS.generate_walk:
            self.batch_size = FLAGS.batch_size
            while ((FLAGS.char_img_n * self.char_embedding_n) % self.batch_size != 0) or (self.batch_size % self.char_embedding_n != 0):
                self.batch_size -= 1
            print('batch_size: {}'.format(self.batch_size))
            if FLAGS.generate_walk:
                self.walk_step = self.batch_size // self.char_embedding_n
                print('walk_step: {}'.format(self.walk_step))
            self._load_dataset()
        else:
            self._setup_inputs()
        self._prepare_generating()

    def _setup_dirs(self):
        """Setup output directories

        If destinations are not existed, make directories like this:
            FLAGS.gan_dir
            ├ generated
            └ random_walking
        """
        self.src_log = os.path.join(FLAGS.gan_dir, 'log')
        self.dst_generated = os.path.join(FLAGS.gan_dir, 'datasets')
        if not os.path.exists(self.dst_generated):
            os.mkdir(self.dst_generated)
        if FLAGS.generate_walk:
            self.dst_walk = os.path.join(FLAGS.gan_dir, 'random_walking')
            if not os.path.exists(self.dst_walk):
                os.makedirs(self.dst_walk)

    def _setup_params(self):
        """Setup paramaters

        To setup GAN, read JSON file and set as attribute (self.~).
        JSON file's path is "FLAGS.gan_dir/log/flags.json".
        """
        with open(os.path.join(self.src_log, 'flags.json'), 'r') as json_file:
            json_dict = json.load(json_file)
        keys = ['chars_type', 'img_width', 'img_height', 'img_dim', 'style_z_size', 'font_h5',
                'style_ids_n', 'arch']
        for key in keys:
            setattr(self, key, json_dict[key])

    def _setup_embedding_chars(self):
        """Setup embedding characters

        Setup generating characters, like alphabets or hiragana.
        """
        self.embedding_chars = set_chars_type(self.chars_type)
        assert self.embedding_chars != [], 'embedding_chars is empty'
        self.char_embedding_n = len(self.embedding_chars)

    def _setup_inputs(self):
        """Setup inputs

        Setup generating inputs, batchsize and others.
        """
        # setting batch_size
        self.batch_size = 5

        # generate char label ,size = batch_size
        self.char_gen_ids = np.random.randint(0,self.char_embedding_n, self.batch_size)

        # generate or setting latent variable
        self.latent = tf.random_uniform((self.batch_size, self.style_z_size), -1, 1)

        sess = tf.InteractiveSession()
        self.latent_numpy = self.latent.eval(session=sess)
        #print(self.latent_numpy)
        sess.close()

        #self.latent = tf.convert_to_tensor(self.latent_numpy)

        self.col_n = self.batch_size
        self.row_n = math.ceil(self.batch_size / self.col_n)

        """
        csv_filename = os.path.join(opt.output_dirpath,"text_information.csv")
        Header = ["Char","latent"]

        with open(csv_filename,'w') as f:
        writer = csv.DictWriter(f,fieldnames=Header)
        writer.writeheader()
        """

    def _load_dataset(self):
        """Load dataset

        Setup dataset.
        """
        self.real_dataset = Dataset(self.font_h5, 'r', self.img_width, self.img_height, self.img_dim)
        self.real_dataset.set_load_data()
        self.alphabet_dict = self.real_dataset.label_to_id

    def _prepare_generating(self):
        """Prepare generating

        Make tensorflow's graph.
        """
        self.z_size = self.style_z_size + self.char_embedding_n

        if self.arch == 'DCGAN':
            generator = GeneratorDCGAN(img_size=(self.img_width, self.img_height),
                                       img_dim=self.img_dim,
                                       z_size=self.z_size,
                                       layer_n=4,
                                       k_size=3,
                                       smallest_hidden_unit_n=64,
                                       is_bn=False)
        elif self.arch == 'ResNet':
            generator = GeneratorResNet(k_size=3, smallest_unit_n=64)

        if FLAGS.generate_walk:
            style_embedding_np = np.random.uniform(-1, 1, (FLAGS.char_img_n // self.walk_step, self.style_z_size)).astype(np.float32)
        else:
            style_embedding_np = np.random.uniform(-1, 1, (self.style_ids_n, self.style_z_size)).astype(np.float32)

        with tf.variable_scope('embeddings'):
            style_embedding = tf.Variable(style_embedding_np, name='style_embedding')


        self.char_ids = tf.placeholder(tf.int32, (self.batch_size,), name='char_ids')

        style_z = self.latent
        char_z = tf.one_hot(self.char_ids, self.char_embedding_n)   

        z = tf.concat([style_z, char_z], axis=1)

        self.generated_imgs = generator(z, is_train=False)

        if FLAGS.gpu_ids == "":
            sess_config = tf.ConfigProto(
                device_count={"GPU": 0},
                log_device_placement=True
            )
        else:
            sess_config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(visible_device_list=FLAGS.gpu_ids)
            )
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())

        if FLAGS.generate_walk:
            var_list = [var for var in tf.global_variables() if 'embedding' not in var.name]
        else:
            var_list = [var for var in tf.global_variables()]
        
        pretrained_saver = tf.train.Saver(var_list=var_list)

        ckpt_filename =  "result.ckpt-7581" 
        ckpt_filepath = os.path.join(self.src_log,ckpt_filename)
        pretrained_saver.restore(self.sess, ckpt_filepath)

        #checkpoint = tf.train.get_checkpoint_state(self.src_log)
        #assert checkpoint, 'cannot get checkpoint: {}'.format(self.src_log)
        #pretrained_saver.restore(self.sess, checkpoint.model_checkpoint_path)
        #pretrained_saver.restore(self.sess, ckpt_filepath)

    def _save_imgs_labels_and_latent(  
                                self, 
                                src_imgs, 
                                labels, 
                                vectors,
                                dst_dir, 
                                iter, 
                                fp):
        """save images

        save each images by each names

        Args:
            src_imgs: Images that will be saved.
            labels  : Character ID used to generate images
            vectors : Multidimensional uniform random number to generate images
            dst_path: Destination path of image.
            iter    : the number of iteration
            fp      : file pointer to save information to .csv
        """
        #concated_img = concat_imgs(src_imgs, self.row_n, self.col_n)
        for i,(src_img,label,vector) in enumerate(zip(src_imgs,labels,vectors)):
            src_img = (src_img + 1.) * 127.5
            if self.img_dim == 1:
                src_img = np.reshape(src_img, (-1, 1 * self.img_height))
            else:
                sec_img = np.reshape(src_img, (-1, 1 * self.img_height, self.img_dim))

            image_dir_path = os.path.join(dst_dir, 'images')
            vector_dir_path = os.path.join(dst_dir, 'vectors')

            if not os.path.exists(image_dir_path):
                os.mkdir(image_dir_path)

            pil_img = Image.fromarray(np.uint8(src_img))
            pil_img.save(os.path.join(image_dir_path, '{}.png'.format(iter+i)))

            if not os.path.exists(vector_dir_path):
                os.mkdir(vector_dir_path)
            
            np.save(os.path.join(vector_dir_path, '{}'.format(iter+i)),vector)

            fp.writerow({"image_id":iter+i,"char_label":self.translater.num2chr(label)})
            

    def generate(self, filename='datasets',gen_num=20):
        """Generate fonts

        Generate fonts from raw data or random input.
        """
        csv_filename = os.path.join(self.dst_generated,"information.csv")
        Header = ["image_id","char_label"]

        with open(csv_filename,'w') as f:
            writer = csv.DictWriter(f,fieldnames=Header)
            writer.writeheader()
            for i in range(gen_num//self.batch_size):
                generated_imgs = self.sess.run(self.generated_imgs,feed_dict={self.char_ids: self.char_gen_ids})
                self._save_imgs_labels_and_latent( 
                                            src_imgs=generated_imgs, 
                                            labels=self.char_gen_ids, 
                                            vectors=self.latent_numpy,
                                            dst_dir=self.dst_generated,
                                            iter=i*self.batch_size,
                                            fp=writer)
                self._setup_inputs()
