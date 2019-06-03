import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

def inception_score(imgs, cuda=True, batch_size=32, resize=True, splits=10):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').cuda()
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = torch.cat(batch)
        batch = batch.cuda()
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar = dset.CIFAR10(root='data/', download=True,
                             transform=transforms.Compose([
                                 transforms.Scale(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
    )

    IgnoreLabelDataset(cifar)

    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(cifar), cuda=True, batch_size=32, resize=True, splits=10))


# # Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import os.path
# import sys
# import tarfile
#
# import numpy as np
# from six.moves import urllib
# import tensorflow as tf
# import glob
# import scipy.misc
# import math
# import sys
#
# MODEL_DIR = '/tmp/imagenet'
# DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# softmax = None
#
# # Call this function with list of images. Each of elements should be a
# # numpy array with values ranging from 0 to 255.
# def get_inception_score(images, splits=10):
#   assert(type(images) == list)
#   assert(type(images[0]) == np.ndarray)
#   assert(len(images[0].shape) == 3)
#   assert(np.max(images[0]) > 10)
#   assert(np.min(images[0]) >= 0.0)
#   inps = []
#   for img in images:
#     img = img.astype(np.float32)
#     inps.append(np.expand_dims(img, 0))
#   bs = 1
#   with tf.Session() as sess:
#     preds = []
#     n_batches = int(math.ceil(float(len(inps)) / float(bs)))
#     for i in range(n_batches):
#         sys.stdout.write(".")
#         sys.stdout.flush()
#         inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
#         inp = np.concatenate(inp, 0)
#         pred = sess.run(softmax, {'ExpandDims:0': inp})
#         preds.append(pred)
#     preds = np.concatenate(preds, 0)
#     scores = []
#     for i in range(splits):
#       part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
#       kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
#       kl = np.mean(np.sum(kl, 1))
#       scores.append(np.exp(kl))
#     return np.mean(scores), np.std(scores)
#
# # This function is called automatically.
# def _init_inception():
#   global softmax
#   if not os.path.exists(MODEL_DIR):
#     os.makedirs(MODEL_DIR)
#   filename = DATA_URL.split('/')[-1]
#   filepath = os.path.join(MODEL_DIR, filename)
#   if not os.path.exists(filepath):
#     def _progress(count, block_size, total_size):
#       sys.stdout.write('\r>> Downloading %s %.1f%%' % (
#           filename, float(count * block_size) / float(total_size) * 100.0))
#       sys.stdout.flush()
#     filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
#     print()
#     statinfo = os.stat(filepath)
#     print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
#   tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
#   with tf.gfile.FastGFile(os.path.join(
#       MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     _ = tf.import_graph_def(graph_def, name='')
#   # Works with an arbitrary minibatch size.
#   with tf.Session() as sess:
#     pool3 = sess.graph.get_tensor_by_name('pool_3:0')
#     ops = pool3.graph.get_operations()
#     for op_idx, op in enumerate(ops):
#         for o in op.outputs:
#             shape = o.get_shape()
#             shape = [s.value for s in shape]
#             new_shape = []
#             for j, s in enumerate(shape):
#                 if s == 1 and j == 0:
#                     new_shape.append(None)
#                 else:
#                     new_shape.append(s)
#             o.set_shape(tf.TensorShape(new_shape))
#     w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
#     logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
#     softmax = tf.nn.softmax(logits)
#
# if softmax is None:
#   _init_inception()
