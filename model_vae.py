import numpy as np
import tensorflow as tf
import tfbox_v4.layers as L
from tfbox_v4.models import Model
from tfbox_v4.config import TF_DTYPE

MIN_FEAT_SIZE = 4
MAX_CHN = 128
MAX_CHN_COND = 128

# append modules
def _append_Act(model, bottom, name, act_config):
    with tf.variable_scope(name):
        if act_config['use_bn']:
            if act_config.get('condition') is None:
                # unconditional batch norm
                tops = model.append(L.BatchNorm(
                    bottom,
                    phase=act_config['phase'],
                    name='bn',
                    update_collection=act_config['update_collection']))
            else:
                # conditional batch norm
                tops = model.append(L.CondBatchNorm(
                    bottom, act_config['condition'],
                    phase=act_config['phase'],
                    name='cond_bn',
                    update_collection=act_config['update_collection']))
        else:
            tops = [bottom]
        tops = model.append(L.Act(tops[0], act_config['args'], name='relu'))
    return tops

def _append_ResBlock(model, bottom, chn, stride, act_config, both_act=False,
        bottleneck=False, name='block', filler=('msra', 0., 1.), kernel_size=3):
    chn_in = bottom.shape.as_list()[-1]
    add_bias = not act_config['use_bn']
    with tf.variable_scope(name):
        tops = _append_Act(model, bottom, 'act_pre', act_config)
        if both_act:
            shortcut_top = tops[-1]
        else:
            shortcut_top = bottom
        if bottleneck:
            chn_out = chn*4
            tops = model.append(
                    L.Conv2d(tops[-1], [1, 1, chn_in, chn], bias=add_bias, name='conv1', filler=filler))
            tops = _append_Act(model, tops[-1], 'act1', act_config)
            tops = model.append(
                    L.Conv2d(tops[-1], [kernel_size, kernel_size, chn, chn], bias=add_bias,
                        stride=stride, pad_size=-1, name='conv2', filler=filler))
            tops = _append_Act(model, tops[-1], 'act2', act_config)
            tops = model.append(
                    L.Conv2d(tops[-1], [1, 1, chn, chn_out], bias=add_bias, name='conv3', filler=filler))
        else:
            chn_out = chn
            tops = model.append(
                    L.Conv2d(tops[-1], [kernel, kernel, chn_in, chn], bias=add_bias,
                        stride=stride, pad_size=-1, name='conv1', filler=filler))
            tops = _append_Act(model, tops[-1], 'act1', act_config)
            tops = model.append(
                    L.Conv2d(tops[-1], [kernel, kernel, chn, chn_out], bias=add_bias,
                        pad_size=-1, name='conv2', filler=filler))
        if chn_in != chn_out or stride > 1:
            # projection to match output shapes
            shortcut_top = model.append(
                    L.Conv2d(shortcut_top, [stride, stride, chn_in, chn_out], bias=add_bias,
                        stride=stride, name='proj', filler=filler))[0]
        tops = model.append(
                L.Add([shortcut_top, tops[-1]]))
    return tops

def _append_DeconvResBlock(model, bottom, chn, stride, act_config, both_act=False,
        bottleneck=False, name='block', filler=('msra', 0., 1.), kernel_size=3):
    chn_in = bottom.shape.as_list()[-1]
    add_bias = not act_config['use_bn']
    with tf.variable_scope(name):
        tops = _append_Act(model, bottom, 'act_pre', act_config)
        if both_act:
            shortcut_top = tops[-1]
        else:
            shortcut_top = bottom
        if bottleneck:
            raise NotImplementedError
        else:
            chn_out = chn
            tops = model.append(
                    L.Conv2d(tops[-1], [kernel_size, kernel_size, chn_in, chn_in], bias=add_bias,
                        pad_size=-1, name='conv1', filler=filler))
            tops = _append_Act(model, tops[-1], 'act1', act_config)
            tops = model.append(
                    L.Deconv2d(tops[-1], [kernel_size, kernel_size, chn_out, chn_in], bias=add_bias,
                        stride=stride, name='deconv2', filler=filler))
        if chn_in != chn_out or stride > 1:
            # projection to match output shapes
            shortcut_top = model.append(
                    L.Deconv2d(shortcut_top, [stride, stride, chn_out, chn_in], bias=add_bias,
                        stride=stride, name='proj', filler=filler))[0]
        tops = model.append(
                L.Add([shortcut_top, tops[-1]]))
    return tops


# model definitions
class Encoder:
    def __init__(self, args, name='encoder'):
        self.name = name
        #
        self.batch_size = args.get('batch_size') or 64
        self.input_chn = args.get('input_chn') or 3
        self.input_size = args.get('input_size') or 256
        self.first_chn = args.get('first_chn') or 16
        self.filler = args.get('filler') or ('msra', 0., 1.)
        self.depth = -1 #args.get('depth') or -1
        self.kernel_size = 3
        self.bottleneck = args.get('bottleneck') or False
        self.n_z = args.get('noise_size') or 128
        max_chn = args.get('max_chn') or MAX_CHN
        #
        # feat_size = [256, 128, 64, 32, 16, 8, 4]
        chn = self.first_chn
        feat_size = self.input_size
        chns = list()
        blocks = list()
        while feat_size >= MIN_FEAT_SIZE:
            chns.append(chn)
            blocks.append(1)
            feat_size /= 2
            chn = min(max_chn, chn*2)
        self.chns = chns
        self.blocks = {
                -1: blocks}
        #
        self._initialized = False

    def build_model(self, inputs, name, phase='train',
            use_bn=True, update_collection=None, has_variance=False, condition=None):
        reuse = self._initialized
        model = Model()
        model.inputs.extend(inputs)
        #
        batch_size = self.batch_size
        input_chn = self.input_chn
        input_size = self.input_size
        filler = self.filler
        depth = self.depth
        kernel_size = self.kernel_size
        bottleneck = self.bottleneck
        n_z = self.n_z
        input_shape = [batch_size, input_size, input_size, input_chn]
        #
        act_config = {
                'phase': phase,
                'use_bn': use_bn,
                'update_collection': update_collection,
                'args': ('ReLU', 0.)}
        #
        with tf.name_scope(name):
            with tf.variable_scope(self.name, reuse=reuse):
                input_tensor = inputs[0]
                tops = [input_tensor]
                input_chn = input_tensor.get_shape().as_list()[-1]
                # conv input
                tops = model.append(
                        L.Conv2d(tops[-1], [kernel_size, kernel_size, input_chn, self.chns[0]],
                            bias=(not use_bn), pad_size=-11, name='conv_input', filler=filler))
                for i, b in enumerate(self.blocks[depth]):
                    chn = self.chns[i]
                    with tf.variable_scope('g%d' % i):
                        for j in xrange(b):
                            stride = 2 if (i > 0 and j == 0) else 1
                            tops = _append_ResBlock(model, tops[-1], chn, stride, act_config,
                                    both_act=(j == 0), bottleneck=bottleneck,
                                    name='b%d' % j, filler=filler, kernel_size=kernel_size)
                tops = _append_Act(model, tops[-1], 'act_top', act_config)
                # avg pool
                pool_shape = tops[-1].get_shape().as_list()
                assert pool_shape[1:] == [MIN_FEAT_SIZE, MIN_FEAT_SIZE, self.chns[-1]], \
                        'Invalid pooling shape (%s)' % str(pool_shape)
                # we skip the pooling layer and directly use two Linear layers to output the mean and log-variance
                #tops = model.append(
                #        L.Pool(tops[0], 'AVG', pool_shape[1], stride=pool_shape[1], name='pool'))
                # fc
                z_mean = model.append(
                        L.Linear(tops[0], n_z, name='fc_z_mean', filler=filler))[0]
                model.outputs.extend([z_mean])
                if has_variance:
                    z_log_var = model.append(
                            L.Linear(tops[0], n_z, name='fc_z_log_var', filler=('uniform', -0.0002, 0.0002)))[0]
                    model.outputs.extend([z_log_var])
                # print structure
                print
                print '========== Setup', name, 'from', input_tensor.name, '=========='
                print
                model.print_model()
        self._initialized = True
        return model

class Decoder:
    def __init__(self, args, name='decoder'):
        self.name = name
        #
        self.batch_size = args.get('batch_size') or 64
        self.input_chn = args.get('input_chn') or 3
        self.input_size = args.get('input_size') or 256
        self.first_chn = args.get('first_chn') or 16
        self.filler = args.get('filler') or ('msra', 0., 1.)
        self.depth = -1 #args.get('depth') or -1
        self.kernel_size = 3
        self.bottleneck = args.get('bottleneck') or False
        self.n_z = args.get('noise_size') or 128
        max_chn = args.get('max_chn') or MAX_CHN
        #
        # feat_size = [4, 8, 16, 32, 64, 128, 256]
        chn = self.first_chn
        feat_size = self.input_size
        chns = list()
        blocks = list()
        while feat_size >= MIN_FEAT_SIZE:
            chns.insert(0, chn)
            blocks.insert(0, 1)
            feat_size /= 2
            chn = min(max_chn, chn*2)
        self.chns = chns
        self.blocks = {
                -1: blocks}
        #
        self._initialized = False

    def build_model(self, inputs, name, phase='train', external_stats=None,
            use_bn=True, update_collection=None, has_variance=False, condition=None):
        reuse = self._initialized
        model = Model()
        model.inputs.extend(inputs)
        #
        batch_size = self.batch_size
        input_chn = self.input_chn
        input_size = self.input_size
        filler = self.filler
        depth = self.depth
        kernel_size = self.kernel_size
        bottleneck = self.bottleneck
        n_z = self.n_z
        input_shape = [batch_size, input_size, input_size, input_chn]
        #
        act_config = {
                'phase': phase,
                'use_bn': use_bn,
                'update_collection': update_collection,
                'args': ('ReLU', 0.),
                'condition': condition}
        #
        with tf.name_scope(name):
            with tf.variable_scope(self.name, reuse=reuse):
                input_tensor = inputs[0]
                tops = [input_tensor]
                # fc latent code
                feat_shape = [batch_size, MIN_FEAT_SIZE, MIN_FEAT_SIZE, self.chns[0]]
                tops = model.append(
                        L.Linear(tops[-1], np.prod(feat_shape[1:]), name='fc_decode', filler=filler))
                tops = model.append(
                        L.Reshape(tops[-1], feat_shape, name='reshape_decode'))
                for i, b in enumerate(self.blocks[depth]):
                    chn = self.chns[i]
                    with tf.variable_scope('g%d' % i):
                        for j in xrange(b):
                            stride = 2 if (i != len(self.blocks[depth]) - 1 and j == b-1) else 1
                            chn = self.chns[i] if (i == len(self.blocks[depth]) - 1 or j != b-1) else self.chns[i+1]
                            tops = _append_DeconvResBlock(model, tops[-1], chn, stride, act_config,
                                    both_act=(j == 0), bottleneck=bottleneck,
                                    name='b%d' % j, filler=filler, kernel_size=kernel_size)
                # no condition for the last batch norm layer
                act_config['condition'] = None
                tops = _append_Act(model, tops[-1], 'act_top', act_config)
                # avg pool
                model.outputs.extend(tops)
                pre_output_shape = tops[-1].get_shape().as_list()
                assert pre_output_shape[1:] == [input_size, input_size, self.chns[-1]], \
                        'Invalid pre-output shape (%s)' % str(pre_output_shape)
                # conv_output
                tops = model.append(
                        L.Conv2d(tops[-1], [kernel_size, kernel_size, self.chns[-1], input_chn], pad_size=-1,
                            name='conv_output'))
                model.outputs.extend(tops)
                # print info
                print
                print '========== Setup', name, 'from', input_tensor.name, '=========='
                print
                model.print_model()
        self._initialized = True
        return model
