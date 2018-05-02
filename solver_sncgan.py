import numpy as np
import os
import os.path as osp
import tensorflow as tf
from time import time
from argparse import ArgumentParser

from tfbox.config import NP_DTYPE, TF_DTYPE
from tfbox.solvers import Solver
from tfbox.layers import Accuracy
from tfbox.util.test_gan_util import gen_seed
from tfbox.dataloaders import RandDataLoaderPrefetch, ImageTransformer

from celeba_dataloader import CelebADataLoader
from model_vae import Encoder as m_dis, Decoder as m_gen
from model_sn_enc import SpecNormEncoder as m_sndis

def print_list(lst):
    for o in lst:
        print o

def print_dict(d):
    for i in d.items():
        print i[0], ":", i[1]

def strf(f):
    s = str(f)
    if '.' in s:
        return s.rstrip('0').rstrip('.')
    return s

class SolverSNCGAN(Solver):
    def __init__(self, args, sess):
        super(SolverSNCGAN, self).__init__(args, sess)

    def setup(self):
        args = self.args
        # args
        mom = args['mom']
        mom2 = args['mom2']
        batch_size = args['batch_size']
        noise_size = args['noise_size']
        cond_size = args['cond_size']
        input_size = args['input_size']
        img_chn = args['img_chn']
        # input shapes
        noise_shape = [batch_size, noise_size]
        img_shape = [batch_size, input_size, input_size, img_chn]
        cond_shape = [batch_size, cond_size]
        # inputs
        lr = tf.placeholder(TF_DTYPE, shape=[], name='lr')
        noise = tf.placeholder(TF_DTYPE, shape=noise_shape, name='noise')
        condition = tf.placeholder(TF_DTYPE, shape=cond_shape, name='condition')
        image = tf.placeholder(TF_DTYPE, shape=img_shape, name='image')
        # model producers
        #args['max_chn'] = 512
        args['input_chn'] = img_chn
        G = m_gen(args, name='G')
        args['noise_size'] = 1
        D = m_sndis(args, name='D_sn')
        D_real_phase = 'train'
        D_update_collection = 'sn'
        args['noise_size'] = noise_size
        #
        for dvid in xrange(1):
            print "~~~~~~~~~~ Device %d ~~~~~~~~~~" % dvid
            with tf.device('/%cpu:%d' % ('c' if args['cpu_only'] else 'g', dvid)):
                #####
                # models
                # gen
                G_fake = G.build_model([noise], 'G_fake', phase='train',
                        update_collection='bn_fake', condition=condition)
                gen_img_before_tanh = G_fake.outputs[-1]
                gen_img = tf.tanh(gen_img_before_tanh, name='gen_img')
                # dis real
                D_real = D.build_model([image], 'D_real', phase=D_real_phase,
                        update_collection=D_update_collection, condition=condition)
                dis_real = D_real.outputs[-1]
                # dis fake
                D_fake = D.build_model([gen_img], 'D_fake', phase='train',
                        update_collection=D_update_collection, condition=condition)
                dis_fake = D_fake.outputs[-1]

                #####
                # losses
                l_D_real = tf.reduce_mean(tf.maximum(1 - dis_real, 0), name='l_D_real')
                l_D_fake = tf.reduce_mean(tf.maximum(1 + dis_fake, 0), name='l_D_fake')
                l_D = tf.add(0.5*l_D_real, 0.5*l_D_fake, name='l_D')
                l_G = tf.reduce_mean(-dis_fake, name='l_G')

                ##########
                # optimizers and update opts
                opt_G = tf.train.AdamOptimizer(lr, mom, mom2)
                opt_D = tf.train.AdamOptimizer(lr, mom, mom2)
                gv_G = opt_G.compute_gradients(l_G, var_list=G_fake.params,
                        colocate_gradients_with_ops=True)
                gv_D = opt_D.compute_gradients(l_D, var_list=D_real.params,
                        colocate_gradients_with_ops=True)
                with tf.control_dependencies([gv[0] for gv in gv_G]):
                    ag_G = opt_G.apply_gradients(gv_G, name='ag_G')
                    update_G = G_fake.get_update_ops('bn_fake')
                with tf.control_dependencies([gv[0] for gv in gv_D]):
                    ag_D = opt_D.apply_gradients(gv_D, name='ag_D')
                    update_D = D_fake.get_update_ops(D_update_collection)

                ##########
                # restore or init
                self.restore_or_init()

                ##########
                # summary
                # inputs
                self.image = image
                self.noise = noise
                self.condition = condition
                self.lr = lr
                # models
                self.G_fake = G_fake
                self.D_real = D_real
                self.D_fake = D_fake
                # outputs
                self.gen_img = gen_img
                self.dis_real = tf.reduce_mean(dis_real)
                self.dis_fake = tf.reduce_mean(dis_fake)
                self.l_D = l_D
                self.l_G = l_G
                # update ops
                self.ag_G = ag_G
                self.update_G = update_G
                self.ag_D = ag_D
                self.update_D = update_D
                # test outputs
                self.test_img = image
                self.test_gen_img = gen_img
                #
                if not args['test_only']:
                    with tf.name_scope('summary'):
                        s_train = list()
                        s_train.append(tf.summary.scalar('dis_real', self.dis_real))
                        s_train.append(tf.summary.scalar('dis_fake', self.dis_fake))
                        s_train.append(tf.summary.scalar('l_D', l_D))
                        self.merged_s_train = tf.summary.merge(s_train)
                        #
                        s_test = list()
                        self.test_img_ph = tf.placeholder(TF_DTYPE, shape=img_shape,
                                name='test_img_ph')
                        self.test_gen_img_ph = tf.placeholder(TF_DTYPE, shape=img_shape,
                                name='test_gen_img_ph')
                        s_test.append(tf.summary.image('test_img',
                            self.get_plottable_data(self.test_img_ph), max_outputs=12))
                        s_test.append(tf.summary.image('test_gen_img',
                            self.get_plottable_data(self.test_gen_img_ph), max_outputs=12))
                        self.merged_s_test = tf.summary.merge(s_test)
                        #
                        self.writer = tf.summary.FileWriter(args['save_dir'], self.sess.graph)

    def get_plottable_data(self, data, minv=0, maxv=255, dtype=tf.uint8):
        return tf.cast(tf.clip_by_value(data, minv, maxv), dtype, name='plottable_data')

    def get_plottable_data_numpy(self, data, minv=0, maxv=255, dtype=np.uint8):
        return np.require(np.clip(data, minv, maxv), dtype)

    def get_dataloader(self, name):
        if self._dataloader.get(name) is None:
            args = self.args
            folder = args['folder']
            names = args['names']
            with_attr = True
            queue_size = args['queue_size']
            nproc = args['nproc']
            if name == 'image':
                blob_names = [self.image.name, self.condition.name]
                blob_shapes = [self.image.shape.as_list(), self.condition.shape.as_list()]
                mean = np.array([127.5, 127.5, 127.5])
                std = np.array([127.5, 127.5, 127.5])
                scale = np.array(0.5)
                tf_image = ImageTransformer({blob_names[0]: blob_shapes[0]})
                tf_image.set_mean(blob_names[0], mean)
                tf_image.set_std(blob_names[0], std)
                tf_image.set_scale(blob_names[0], scale)
                tf_image.set_center(blob_names[0], True)
                tf_image.set_mirror(blob_names[0], False)
                dl = CelebADataLoader(folder, names, with_attr, queue_size,
                        transformer=tf_image, seed=self.next_seed())
                dl.add_prefetch_process(name, blob_names, blob_shapes, nproc, seeds=self.next_seed(nproc))
            elif name == 'noise':
                blob_name = self.noise.name
                blob_shape = self.noise.get_shape().as_list()
                dl = RandDataLoaderPrefetch(queue_size, seed=self.next_seed())
                dl.add_prefetch_process(blob_name, blob_shape, nproc=1, seeds=self.next_seed(1))
            else:
                dl = None
            self._dataloader[name] = dl
        return self._dataloader.get(name)

    def _step_gen(self, lr):
        dl_image = self.get_dataloader('image')
        dl_noise = self.get_dataloader('noise')
        noise_name = self.noise.name
        #
        _, (image, condition) = dl_image._get_data('image')
        noise = dl_noise._get_data(noise_name)
        #
        fetch_list = [self.ag_G] + self.update_G
        feed_dict = {
                self.image: image,
                self.condition: condition,
                self.noise: noise,
                self.lr: lr}
        fetch_val = self.sess.run(fetch_list, feed_dict)
    
    def _step_dis(self, lr):
        dl_image = self.get_dataloader('image')
        dl_noise = self.get_dataloader('noise')
        noise_name = self.noise.name
        #
        _, (image, condition) = dl_image._get_data('image')
        noise = dl_noise._get_data(noise_name)
        #
        fetch_dict = {
                'l_D': self.l_D,
                'dis_real': self.dis_real,
                'dis_fake': self.dis_fake,
                'ag_D': self.ag_D,
                'update_D': self.update_D,
                'sum': self.merged_s_train}
        feed_dict = {
                self.image: image,
                self.condition: condition,
                self.noise: noise,
                self.lr: lr}
        fetch_val = self.sess.run(fetch_dict, feed_dict)
        return fetch_val

    def step(self, lr):
        for _ in xrange(self.args['d_iter']):
            ret = self._step_dis(lr)
        self._step_gen(lr)
        return ret

    def train(self):
        args = self.args
        batch_size = args['batch_size']
        target_batch_size = args['target_batch_size']
        lr = NP_DTYPE(args['lr'])
        fixed_batchids = None
        fixed_noise = None
        saver = self.get_saver()

        start_time = time()
        for itr in xrange(args['mxit'] + 1):
            if itr >= args['lr_update_after'] and (itr - args['lr_update_after']) % args['lr_update_every'] == 0:
                lr *= args['lr_decay']
            # test
            if args['tsit'] > 0 and itr % args['tsit'] == 0 and (itr > 0 or args['test_first']):
                ret = self.test(itr, fixed_noise, fixed_batchids)
                fixed_batchids = ret['batchids']
                fixed_noise = ret['noise']
                self.writer.add_summary(ret['summary'], itr * batch_size/target_batch_size)
                end_time = time()
                print 'Test [%d](%.2f)' % (itr, end_time-start_time)
                start_time = end_time
            # snapshot
            if itr % args['ssit'] == 0:
                self.save(self.sess, saver, self.save_dir, global_step = itr)
                end_time = time()
                print 'Snapshot(%.2f)' % (end_time - start_time)
                start_time = end_time
            # train
            ret = self.step(lr)
            self.writer.add_summary(ret['sum'], itr*batch_size/target_batch_size)
            if itr > 0 and itr % args['dpit'] == 0:
                end_time = time()
                print '[%d](%.2f)' % (itr, end_time - start_time)
                print 'lr: %.4e|' % lr,
                print 'l_D: %.4e|' % ret['l_D'],
                print 'd_real: %.4e|' % ret['dis_real'],
                print 'd_fake: %.4e|' % ret['dis_fake']
                start_time = end_time

    def test(self, itr, noise, batchids):
        args = self.args
        #
        image_name = self.image.name
        image_shape = self.image.shape.as_list()
        image = np.zeros(image_shape, dtype=NP_DTYPE)
        condition_name = self.condition.name
        condition_shape = self.condition.shape.as_list()
        condition = np.zeros(condition_shape, dtype=NP_DTYPE)
        #
        dl_image = self.get_dataloader('image')
        batchids = dl_image.fill_input([image, condition], [image_name, condition_name], batchids, data_name='image')
        if noise is None:
            noise_name = self.noise.name
            noise_shape = self.noise.shape.as_list()
            dl_noise = self.get_dataloader('noise')
            noise = dl_noise._get_data(noise_name)
        #
        fetch_list = [self.test_img, self.test_gen_img]
        #
        fetch_val = self.sess.run(fetch_list,
                {self.image: image,
                    self.condition: condition,
                    self.noise: noise})
        #
        test_img = fetch_val[0] * 127.5 + 127.5
        test_gen_img = fetch_val[1] * 127.5 + 127.5
        #
        if self.args['test_only']:
            test_img = self.get_plottable_data_numpy(test_img)
            test_gen_img = self.get_plottable_data_numpy(test_gen_img)
            #
            output_file = osp.join(self.save_dir, 'results_test_only%d.npz' % (itr if itr is not None else 0))
            np.savez(output_file, img=test_img, gen_img=test_gen_img, condition=condition)
            return {}
        else:
            feed_dict = { 
                self.test_img_ph: test_img,
                self.test_gen_img_ph: test_gen_img}
            summary = self.sess.run(self.merged_s_test, feed_dict)
            return {'batchids': batchids,
                    'noise': noise,
                    'summary': summary}

    @classmethod
    def adjust_params(cls, args):
        batch_size = args.batch_size
        target_batch_size = args.target_batch_size
        bot = batch_size*1./target_batch_size
        tob = target_batch_size*1./batch_size
        #
        args.lr *= bot
        args.mxit = int(args.mxit*tob)
        args.dpit = int(args.dpit*tob)
        args.tsit = int(args.tsit*tob)
        args.ssit = int(args.ssit*tob)
        #
        args.lr_update_after = int(args.lr_update_after*tob)
        args.lr_update_every = int(args.lr_update_every*tob)
        #
        args.save_dir = osp.join(args.save_dir,
                "%s_lr%s_bs%dt%d_ns%d_fc%d_maxc%d_d%dg%d" % (
                    args.name, strf(args.lr),
                    args.batch_size, args.target_batch_size,
                    args.noise_size, args.first_chn, args.max_chn,
                    args.d_iter, args.g_iter))
        #
        if args.no_bn:
            args.save_dir += '_nobn'
        else:
            args.save_dir += '_usebn'
        #
        if args.gstd == -1:
            args.filler = ('msra', 0., 1.)
            args.save_dir += '_msra'
        else:
            args.filler = ('gaussian', 0., args.gstd)
            args.save_dir += '_gstd%s' % strf(args.gstd)

    @classmethod
    def parse_args(cls, ps=None):
        if ps is None:
            ps = cls.get_parser()
        args = ps.parse_args()
        cls.adjust_params(args)
        return vars(args)

    @classmethod
    def get_parser(cls, ps=None):
        if ps is None:
            ps = ArgumentParser()
        # dataset
        ps.add_argument('--folder', type=str, default='/DATA/sw015/img_align_celeba')
        ps.add_argument('--names', type=str, default='/DATA/sw015/img_align_celeba/list_attr_celeba.txt')
        ps.add_argument('--queue_size', type=int, default=2)
        ps.add_argument('--nproc', type=int, default=2)
        # network
        ps.add_argument('--batch_size', type=int, default=64)
        ps.add_argument('--cond_size', type=int, default=40)
        ps.add_argument('--target_batch_size', type=int, default=64)
        ps.add_argument('--input_size', type=int, default=64)
        ps.add_argument('--img_chn', type=int, default=3)
        ps.add_argument('--first_chn', type=int, default=32)
        ps.add_argument('--max_chn', type=int, default=512)
        ps.add_argument('--noise_size', type=int, default=128)
        ps.add_argument('--depth', type=int, default=-1)
        ps.add_argument('--gstd', type=float, default=-1)
        ps.add_argument('--bottleneck', action='store_true', default=False)
        ps.add_argument('--no_bn', action='store_true', default=False)
        # solver
        ps.add_argument('--name', type=str, default='sn_cgan')
        ps.add_argument('--lr', type=float, default=0.00002)
        ps.add_argument('--mom', type=float, default=0.5)
        ps.add_argument('--mom2', type=float, default=0.99)
        ps.add_argument('--lr_update_after', type=int, default=64000)
        ps.add_argument('--lr_update_every', type=int, default=32000)
        ps.add_argument('--lr_decay', type=float, default=0.1)
        ps.add_argument('--global_step', type=int, default=-1, help='the iteration of model to resume')
        ps.add_argument('--save_dir', type=str, default='results')
        ps.add_argument('--mxit', type=int, default=64000)
        ps.add_argument('--dpit', type=int, default=20)
        ps.add_argument('--tsit', type=int, default=200)
        ps.add_argument('--ssit', type=int, default=8000)
        ps.add_argument('--max_to_keep', type=int, default=10)
        ps.add_argument('--test_first', action='store_true', default=False)
        ps.add_argument('--test_only', action='store_true', default=False)
        ps.add_argument('--d_iter', type=int, default=5, help='# updates of d per step')
        ps.add_argument('--g_iter', type=int, default=1, help='# updates of g per step')
        ps.add_argument('--cpu_only', action='store_true', default=False)
        #
        return ps

def main():
    import sys, traceback
    args = SolverSNCGAN.parse_args()
    print_dict(args)

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        solver = SolverSNCGAN(args, sess)
        print "Before train/test"
        try:
            if args['test_only']:
                for i in xrange(10):
                    solver.test(-1-i, None, None)
            else:
                solver.train()
        except KeyboardInterrupt:
            print "Interrupted by user"
        except:
            print "Unexpected error:", sys.exc_info()
            traceback.print_exc()

if __name__ == '__main__':
    main()
