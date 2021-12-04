from collections import namedtuple
from models import generator_resnet, discriminator
from utils import *
from loss_utils import *
from ops import *
import time
import matplotlib.pyplot as plt
from glob import glob



class AUGAN(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.conf_lambda = args.conf_lambda
        self.dataset_dir = args.dataset_dir
        self.n_d = args.n_d
        self.n_scale = args.n_scale
        self.ndf = args.ndf
        self.load_size = args.load_size
        self.fine_size = args.fine_size
        self.generator = generator_resnet
        self.discriminator = discriminator
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
            self.criterionGAN_list = mae_criterion_list
        else:
            self.criterionGAN = sce_criterion
            self.criterionGAN_list = sce_criterion_list

        self.use_uncertainty = args.use_uncertainty

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf // args.n_d, args.output_nc,
                                      args.phase == 'train'))
        self.save_conf = args.save_conf
        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size * 2,
                                                     self.input_c_dim + self.output_c_dim], name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        A_label = np.zeros([1, 1, 1, 2], dtype=np.float32)
        B_label = np.zeros([1, 1, 1, 2], dtype=np.float32)
        A_label[:, :, :, 0] = 1.0
        B_label[:, :, :, 1] = 1.0
        self.A_label = tf.convert_to_tensor(A_label)
        self.B_label = tf.convert_to_tensor(B_label)

        self.fake_B, self.rec_realA, self.realA_percep, self.transA_percep, self.pred_confA = self.generator(
            self.real_A, self.options, transfer=True, reuse=False, name="generatorA2B")
        self.fake_A_, self.rec_fakeB, self.fakeB_percep, _, _ = self.generator(self.fake_B, self.options,
                                                                               transfer=False, reuse=False,
                                                                               name="generatorB2A")
        self.fake_A, self.rec_realB, self.realB_percep, _, _ = self.generator(self.real_B, self.options, transfer=False,
                                                                              reuse=True, name="generatorB2A")
        self.fake_B_, self.rec_fakeA, self.fakeA_percep, self.trans_fakeA_percep, _ = self.generator(self.fake_A,
                                                                                                     self.options,
                                                                                                     transfer=True,
                                                                                                     reuse=True,
                                                                                                     name="generatorA2B")

        self.g_adv_total = 0.0
        self.g_adv = 0.0
        self.g_adv_rec = 0.0
        self.g_adv_recfake = 0.0

        self.percep_loss = tf.reduce_mean(
            tf.abs(tf.reduce_mean(self.transA_percep, axis=3) - tf.reduce_mean(self.fakeB_percep, axis=3))) \
                           + tf.reduce_mean(
            tf.abs(tf.reduce_mean(self.realB_percep, axis=3) - tf.reduce_mean(self.fakeA_percep, axis=3)))

        for i in range(self.n_d):
            self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name=str(i) + "_discriminatorB")
            self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name=str(i) + "_discriminatorA")

            self.g_adv_total += self.criterionGAN_list(self.DA_fake,
                                                       get_ones_like(self.DA_fake)) + self.criterionGAN_list(
                self.DB_fake, get_ones_like(self.DB_fake))


            self.g_adv += self.criterionGAN_list(self.DA_fake, get_ones_like(self.DA_fake)) + self.criterionGAN_list(
                self.DB_fake, get_ones_like(self.DB_fake))

        self.g_loss_a2b = self.criterionGAN_list(self.DB_fake, get_ones_like(self.DB_fake)) \
                          + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
                          + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_b2a = self.criterionGAN_list(self.DA_fake, get_ones_like(self.DA_fake)) \
                          + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
                          + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)

        self.g_A_recon_loss = self.L1_lambda * abs_criterion(self.rec_realA, self.real_A)
        self.g_B_recon_loss = self.L1_lambda * abs_criterion(self.rec_realB, self.real_B)
        if self.use_uncertainty:
            self.g_A_cycle_loss = self.conf_lambda * conf_criterion(self.real_A, self.fake_A_, self.pred_confA)
        else:
            self.g_A_cycle_loss = self.L1_lambda * abs_criterion(self.real_A, self.fake_A_)
        self.g_B_cylce_loss = self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)

        self.g_loss = self.g_adv_total \
                      + self.g_A_recon_loss \
                      + self.g_B_recon_loss \
                      + self.g_A_cycle_loss \
                      + self.g_B_cylce_loss \
                      + self.percep_loss


        self.g_rec_real = abs_criterion(self.rec_realA, self.real_A) + abs_criterion(self.rec_realB, self.real_B)
        self.g_rec_cycle = abs_criterion(self.real_A, self.fake_A_) + abs_criterion(self.real_B, self.fake_B_)

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [self.batch_size, self.image_size, self.image_size * 2, self.output_c_dim],
                                            name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [self.batch_size, self.image_size, self.image_size * 2, self.output_c_dim],
                                            name='fake_B_sample')
        self.rec_A_sample = tf.placeholder(tf.float32,
                                           [self.batch_size, self.image_size, self.image_size * 2, self.output_c_dim],
                                           name='rec_A_sample')
        self.rec_B_sample = tf.placeholder(tf.float32,
                                           [self.batch_size, self.image_size, self.image_size * 2, self.output_c_dim],
                                           name='rec_B_sample')
        self.rec_fakeA_sample = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size * 2,
                                                            self.output_c_dim], name='rec_fakeA_sample')
        self.rec_fakeB_sample = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size * 2,
                                                            self.output_c_dim], name='rec_fakeB_sample')

        self.d_loss_item = []
        self.d_loss_item_rec = []
        self.d_loss_item_recfake = []

        for i in range(self.n_d):
            self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name=str(i) + "_discriminatorB")
            self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name=str(i) + "_discriminatorA")
            self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True,
                                                     name=str(i) + "_discriminatorB")
            self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True,
                                                     name=str(i) + "_discriminatorA")
            self.db_loss_real = self.criterionGAN_list(self.DB_real, get_ones_like(self.DB_real))
            self.db_loss_fake = self.criterionGAN_list(self.DB_fake_sample, get_zeros_like(self.DB_fake_sample))
            self.db_loss = (self.db_loss_real * 0.5 + self.db_loss_fake * 0.5)
            self.da_loss_real = self.criterionGAN_list(self.DA_real, get_ones_like(self.DA_real))
            self.da_loss_fake = self.criterionGAN_list(self.DA_fake_sample, get_zeros_like(self.DA_fake_sample))
            self.da_loss = (self.da_loss_real * 0.5 + self.da_loss_fake * 0.5)
            self.d_loss = (self.da_loss + self.db_loss)
            self.d_loss_item.append(self.d_loss)

        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )

        self.test_A = tf.placeholder(tf.float32,
                                     [self.batch_size, self.image_size, self.image_size * 2, self.input_c_dim],
                                     name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [self.batch_size, self.image_size, self.image_size * 2, self.output_c_dim],
                                     name='test_B')

        self.testB, self.rec_testA, self.testA_percep, self.trans_testA_percep, self.test_pred_confA = self.generator(
            self.test_A, self.options, transfer=True, reuse=True, name="generatorA2B")
        self.rec_cycle_A, self.refine_testB, self.testB_percep, _, _ = self.generator(self.testB, self.options,
                                                                                      transfer=False, reuse=True,
                                                                                      name="generatorB2A")

        self.testA, self.rec_testB, _, _, _ = self.generator(self.test_B, self.options, transfer=False, reuse=True,
                                                             name="generatorB2A")
        self.rec_cycle_B, self.refine_testA, _, _, _ = self.generator(self.testA, self.options, True, True,
                                                                      name="generatorA2B")

        t_vars = tf.trainable_variables()

        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.p_vars = [var for var in t_vars if 'percep' in var.name]
        self.d_vars_item = []
        for i in range(self.n_d):
            self.d_vars = [var for var in t_vars if str(i) + '_discriminator' in var.name]
            self.d_vars_item.append(self.d_vars)

    def train(self, args):

        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

        ### generator
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        ### translation
        self.d_optim_item = []
        for i in range(self.n_d):
            self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
                .minimize(self.d_loss_item[i], var_list=self.d_vars_item[i])
            self.d_optim_item.append(self.d_optim)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(os.path.join(args.checkpoint_dir, "logs"), self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
            lr = args.lr if epoch < args.epoch_step else args.lr * (args.epoch - epoch) / (args.epoch - args.epoch_step)

            for idx in range(0, batch_idxs):
                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                batch_images = [load_train_data(batch_file, args.load_size, args.fine_size) for batch_file in
                                batch_files]
                batch_images = np.array(batch_images).astype(np.float32)
                # Update G network and record fake outputs
                fake_A, fake_B, rec_A, rec_B, rec_fake_A, rec_fake_B, _, g_loss, gan_loss, percep, g_adv, g_A_recon_loss, g_B_recon_loss, g_A_cycle_loss, g_B_cycle_loss, summary_str = self.sess.run(
                    [self.fake_A, self.fake_B, self.rec_realA, self.rec_realB, self.rec_fakeA, self.rec_fakeB,
                     self.g_optim, self.g_loss, self.g_adv_total, self.percep_loss, self.g_adv,
                     self.g_A_recon_loss, self.g_B_recon_loss, self.g_A_cycle_loss, self.g_B_cylce_loss,
                     self.g_sum],
                    feed_dict={self.real_data: batch_images, self.lr: lr})
                self.writer.add_summary(summary_str, counter)
                [fake_A, fake_B] = self.pool([fake_A, fake_B])

                # Update D network
                loss_print = []
                for i in range(self.n_d):
                    _, d_loss, d_sum = self.sess.run(
                        [self.d_optim_item[i], self.d_loss_item[i], self.d_sum],
                        feed_dict={self.real_data: batch_images,
                                   self.fake_A_sample: fake_A,
                                   self.fake_B_sample: fake_B, self.lr: lr})

                    loss_print.append(d_loss)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f g_loss: %4.4f gan:%4.4f adv:%4.4f g_percep:%4.4f " % (
                    epoch, idx, batch_idxs, time.time() - start_time, g_loss, gan_loss, g_adv, percep)))

                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, args.save_freq) == 2:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        np.random.shuffle(dataA)
        np.random.shuffle(dataB)
        batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size]))
        sample_images = [load_train_data(batch_file, self.load_size, self.fine_size, is_testing=True) for batch_file in
                         batch_files]
        sample_images = np.array(sample_images).astype(np.float32)

        fake_A, fake_B = self.sess.run(
            [self.fake_A, self.fake_B],
            feed_dict={self.real_data: sample_images}
        )
        real_A = sample_images[:, :, :, :3]
        real_B = sample_images[:, :, :, 3:]



        merge_A = np.concatenate([real_B, fake_A], axis=2)
        merge_B = np.concatenate([real_A, fake_B], axis=2)
        check_folder('./{}/{:02d}'.format(sample_dir, epoch))
        save_images(merge_A, [self.batch_size, 1],
                    './{}/{:02d}/A_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(merge_B, [self.batch_size, 1],
                    './{}/{:02d}/B_{:04d}.jpg'.format(sample_dir, epoch, idx))

    def test(self, args):

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        out_var, refine_var, in_var, rec_var, cycle_var, percep_var, conf_var = (
            self.testB, self.refine_testB, self.test_A, self.rec_testA, self.rec_cycle_A, self.testA_percep,
            self.test_pred_confA) if args.which_direction == 'AtoB' else (
            self.testA, self.refine_testA, self.test_B, self.rec_testB, self.rec_cycle_B, self.testB_percep,
            self.test_pred_confA)
        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(args.test_dir,
                                      '{0}_{1}'.format(args.which_direction, os.path.basename(sample_file)))
            conf_path = os.path.join(args.conf_dir,
                                     '{0}_{1}'.format(args.which_direction, os.path.basename(sample_file)))

            fake_img, = self.sess.run([out_var], feed_dict={in_var: sample_image})
            merge = np.concatenate([sample_image, fake_img], axis=2)
            save_images(merge, [1, 1], image_path)

            if args.save_conf:

                if args.which_direction == 'AtoB':
                    pass
                else:
                    raise Exception('--conf map only can be estimated in AtoB direction')

                conf_img = self.sess.run(conf_var, feed_dict={in_var: sample_image})
                conf_img_sq = np.squeeze(conf_img)
                plt.imshow(conf_img_sq, cmap='plasma', interpolation='nearest', alpha=1.0)
                plt.savefig(conf_path)