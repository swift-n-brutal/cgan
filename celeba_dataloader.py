# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:31:38 2017

@author: shiwu_001
"""

import numpy as np
import lmdb
import caffe.proto as caproto
from multiprocessing import Queue, Process
import io
import zlib
import os.path as osp
from scipy import misc

from tfbox_v4.config import NP_DTYPE as DTYPE
from tfbox_v4.dataloaders import ImageTransformer

import signal
class Handler:
    def __init__(self, idx=-1):
        self.alive = True
        self.idx = idx
        signal.signal(signal.SIGINT, self.on_interruption)
        signal.signal(signal.SIGTERM, self.on_termination)

    def on_interruption(self, sig, frame):
        pass
        #print self.idx, 'Got SIGINT'

    def on_termination(self, sig, frame):
        #print self.idx, 'Got SIGTERM'
        self.alive = False

class CelebADataLoader(object):
    def __init__(self, folder, names, with_attr=True, queue_size=4, seed=None,
                 transformer=None):
        self.folder = folder
        self.with_attr = with_attr
        self.names, self.attrs = self._init_name_list(names)
        self.queue_size = queue_size
        self.rand = np.random.RandomState(seed)
        self.transformer = transformer
        # process to sample batchid
        self.batchids_queue_size = queue_size
        self.batchids_queue_ = dict()
        self.batchids_processes = list()
        # processes to load data
        self.data_queue_size = queue_size
        self.data_queue_ = dict()
        self.worker_processes = list()
    
    def _init_name_list(self, names):
        fnames = file(names, 'r')
        name_list = []
        attr_list = []
        if self.with_attr:
            for line in fnames:
                line = line.rstrip('\n')
                line = line.rstrip('\r')
                name, attr_str = line.split(' ', 1)
                attr_arr = np.fromstring(attr_str, dtype=np.byte, sep=' ')
                name_list.append(name)
                attr_list.append(attr_arr)
        else:
            for line in fnames:
                line = line.rstrip('\n')
                line = line.rstrip('\r')
                name_list.append(line)
        self.n_images = len(name_list)
        print self.n_images, 'images in total'
        return name_list, attr_list

    def _load_batch(self, blobs, blob_names, batchids):
        blob_name = blob_names[0] # assuming the first blob is for image
        for i, index in enumerate(batchids):
            img_path = osp.join(self.folder, self.names[index])
            img = misc.imread(img_path, mode='RGB')
            if self.transformer is not None:
                img = self.transformer.process(blob_name, img)
            else:
                img = img.astype(DTYPE) /127.5 - 1.
            assert img.shape == blobs[0].shape[1:], \
                    'image shape is not equal to blob shape: {} vs {}'.format(img.shape, blobs[0].shape[1:])
            blobs[0][i,...] = img[...]
            if self.with_attr and len(blobs) == 2:
                blobs[1][i,...] = self.attrs[index][...]

    def _get_data(self, data_name):
        dq = self.data_queue_.get(data_name)
        assert dq is not None, 'No such blob specified: %s' % data_name
        return dq.get()

    def _get_batchids_queue(self, data_name, batchsize, seed=None):
        if self.batchids_queue_.get(data_name) is None:
            _queue = Queue(self.batchids_queue_size)
            _process = Process(name='[%s]batchids' % data_name, target=self.__class__._batchids_process,
                    args=(np.random.RandomState(seed), self.n_images, batchsize, _queue))
            _process.start()
            self.batchids_processes.append(_process)
            self.batchids_queue_[data_name] = _queue
        return self.batchids_queue_[data_name]
    
    @classmethod
    def _batchids_process(cls, rand, n_images, batchsize, batchids_queue):
        handler = Handler(0)
        while handler.alive:
            batchids_queue.put(rand.choice(n_images, size=batchsize, replace=False))
        batchids_queue.close()
        
    @classmethod
    def _worker_process(cls, blob_names, data_shapes, data_queue, batchids_queue,
                        transformer, folder, names, attrs, with_attr, seed):
        handler = Handler(seed)
        # independent random seed
        if transformer is not None:
            transformer.rand = np.random.RandomState(seed)
        #
        prefetch_data = list()
        for ds in data_shapes:
            prefetch_data.append(np.zeros(ds, dtype=DTYPE))
        blob_name = blob_names[0] # assuming the first blob is for image
        while handler.alive:
            batchids = batchids_queue.get()
            for i, index in enumerate(batchids):
                img_path = osp.join(folder, names[index])
                img = misc.imread(img_path, mode='RGB')
                if transformer is not None:
                    img = transformer.process(blob_name, img)
                else:
                    img = img.astype(DTYPE) /127.5 - 1.
                assert img.shape == prefetch_data[0].shape[1:], \
                        'image shape is not equal to blob shape: {} vs {}'.format(img.shape, prefetch_data[0].shape[1:])
                prefetch_data[0][i,...] = img[...]
                if with_attr and len(prefetch_data) == 2:
                    prefetch_data[1][i,...] = attrs[index][...]
            data_queue.put((batchids, tuple(d.copy() for d in prefetch_data)))
        data_queue.close()
            
    def set_transformer(self, transformer):
        self.transformer = transformer
    
    def add_prefetch_process(self, data_name, blob_names, data_shapes, nproc=1, seeds=None):
        batchids_queue = self._get_batchids_queue(data_name, data_shapes[0][0])
        assert self.data_queue_.get(data_name) is None, "data queue [%s] already exsits" % data_name
        data_queue = Queue(self.data_queue_size)
        self.data_queue_[data_name] = data_queue
        for i in xrange(nproc):
            if type(seeds) is list and len(seeds) == nproc:
                seed = seeds[i]
            else:
                seed = None
            wp = Process(name='[%s]%d' % (data_name, i), target=self.__class__._worker_process,
                         args=(blob_names, data_shapes, data_queue, batchids_queue,
                               self.transformer, self.folder, self.names, self.attrs, self.with_attr, seed))
            wp.start()
            self.worker_processes.append(wp)
            
    def clean_and_close(self):
        from Queue import Empty
        # first terminate all worker processes
        for wp in self.worker_processes:
            wp.terminate()
        #
        for name, dq in self.data_queue_.items():
            try:
                while True:
                    temp = dq.get(timeout=1)
            except Empty:
                pass
        for dq in self.data_queue_.values():
            dq.close()
            dq.join_thread()
            print "Closed data_queue:", name
        #
        for wp in self.worker_processes:
            while wp.is_alive():
                wp.join(timeout=1)
            print "Joined process:", wp.name
        # terminate batchids process
        for bp in self.batchids_processes:
            bp.terminate()
        #
        for name, bq in self.batchids_queue_.items():
            try:
                while True:
                    bq.get(timeout=1)
            except Empty:
                pass
        for bq in self.batchids_queue_.values():
            bq.close()
            bq.join_thread()
            print "Closed batchids_queue:", name
        #
        for bp in self.batchids_processes:
            bp.join()
            print "Joined process:", bp.name
    
    def __del__(self):
        self.clean_and_close()
        print 'Closed prefetch processes.'
        
    def fill_input(self, blobs, blob_names, batchids, data_name=None):
        assert len(blobs) == 1 or (len(blobs) == 2 and self.with_attr)
        assert len(blob_names) == 1 or (len(blobs) == 2 and self.with_attr)
        if batchids is None:
            batchids, data = self._get_data(data_name)
            blobs[0][...] = data[0][...]
            if len(blobs) == 2:
                blobs[1][...] = data[1][...]
        else:
            self._load_batch(blobs, blob_names, batchids)
        return batchids
            
def get_plottable_data(data, data_format="CHW"):
    data = np.clip(np.round(data), 0, 255).astype(np.uint8)
    if data_format == "CHW":
        data = data.swapaxes(0,1).swapaxes(1,2)
    return data
    

def main():
    import time
    import sys, traceback
    print "Test"
    queue_size = 4
    nproc = 2
    folder = "/home/sw015/data/img_align_celeba"
    names = "/home/sw015/data/img_align_celeba/names_with_attr.txt"
    name_data = 'data'
    name_labels = 'attrs'
    batchsize = 64
    chn = 3
    data_size = 64
    n_attr = 40
    blob_data = np.zeros([batchsize, data_size, data_size, chn], dtype=DTYPE)
    blob_labels = np.zeros([batchsize, 40], dtype=DTYPE)
    scale = np.array(0.5)
    mean = np.array([127.5, 127.5, 127.5])
    std = np.array([128, 128, 128])
    batchids = np.arange(64) # None
    # train transformer
    tf_train = ImageTransformer({name_data: [batchsize, data_size, data_size, chn]})
    tf_train.set_mean(name_data, mean)
    tf_train.set_std(name_data, std)
    tf_train.set_scale(name_data, scale)
    tf_train.set_center(name_data, True)
    tf_train.set_mirror(name_data, True)
    cdl_train = CelebADataLoader(folder, names, True, queue_size, transformer=tf_train)
    cdl_train.add_prefetch_process('train', [name_data, name_labels], [blob_data.shape, blob_labels.shape], nproc)
    #
    try:
        #plt.figure(1)
        for i in xrange(10):
            batchids = cdl_train.fill_input([blob_data, blob_labels],
                                 [name_data, name_labels], batchids=batchids, data_name='train')
            im = tf_train.deprocess(name_data, blob_data[0,...])
            #im = get_plottable_data(im)
            print "Read", batchids,
            print "shape", blob_data.shape
            print "attr", blob_labels[0,...]
            time.sleep(2)
            np.savez('celeba.npz', images=blob_data, attrs=blob_labels)
            break
            #plt.imshow(im)
            #plt.waitforbuttonpress()
            #plt.close()
    except KeyboardInterrupt:
        print "Interrupted by user"
    except:
        print "Unexpected error:", sys.exc_info()
        traceback.print_exc()
    
if __name__ == "__main__":
    main()
