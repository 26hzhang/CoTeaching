import os
import errno
import numpy as np
import codecs
import pickle
import gzip
from six.moves import urllib
from data.utils import noisify


class MNIST:
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt`` and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``, otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
    """
    urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
    raw_folder = 'raw'
    processed_folder = 'processed-tf'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, download=False, noise_type=None, noise_rate=0.2, random_state=0):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.dataset = 'mnist'
        self.noise_type = noise_type

        if download:
            self.download()

        if not self._check_exists():
            os.makedirs(os.path.join(self.root, self.processed_folder))
            self.processing()

        if self.train:
            with open(os.path.join(self.root, self.processed_folder, self.training_file), mode="rb") as handle:
                self.train_data, self.train_labels = pickle.load(handle)
                self.train_data = self.train_data.astype(dtype=np.float32) / 255.0
            if noise_type != 'clean':
                self.train_labels = np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                print(self.train_labels.shape)
                self.train_noisy_labels, self.actual_noise_rate = noisify(train_labels=self.train_labels,
                                                                          noise_type=noise_type,
                                                                          noise_rate=noise_rate,
                                                                          random_state=random_state)
                self.train_noisy_labels = [i[0] for i in self.train_noisy_labels]
                _train_labels = [i[0] for i in self.train_labels]
                self.noise_or_not = np.transpose(self.train_noisy_labels) == np.transpose(_train_labels)
                print("noise or not: ", self.noise_or_not.shape)
        else:
            with open(os.path.join(self.root, self.processed_folder, self.test_file), mode="rb") as handle:
                self.test_data, self.test_labels = pickle.load(handle)
                self.test_data = self.test_data.astype(dtype=np.float32) / 255.0

    def __getitem__(self, index):
        if self.train:
            if self.noise_type != 'clean':
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target, index

    def batch_patcher(self, batch_size=128, drop_last=False):
        if self.train:
            data = np.copy(self.train_data)
            if self.noise_type != "clean":
                labels = np.copy(self.train_noisy_labels)
            else:
                labels = np.copy(self.train_labels)
            ids = np.asarray(list(range(len(self.train_data))))
            p = np.random.permutation(len(data))
            data, labels, ids = data[p], labels[p], ids[p]
        else:
            data = np.copy(self.test_data)
            labels = np.copy(self.test_labels)
            ids = np.asarray(list(range(len(self.test_data))))
        batches = []
        for start in range(0, len(data), batch_size):
            batch_data = data[start: start + batch_size]
            batch_labels = labels[start: start + batch_size]
            batch_ids = ids[start: start + batch_size]
            batches.append((batch_data, batch_labels, batch_ids))
        if len(batches[-1][0]) < batch_size and drop_last:
            batches = batches[:-1]
        return batches

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def _check_raw_file_exists(self):
        return os.path.isdir(os.path.join(self.root, self.raw_folder)) and \
               os.listdir(os.path.join(self.root, self.raw_folder))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        if self._check_raw_file_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

    def processing(self):
        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as handle:
            pickle.dump(training_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as handle:
            pickle.dump(test_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return np.reshape(parsed, newshape=(length,)).astype(dtype=np.int64)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return np.reshape(parsed, newshape=(length, num_rows, num_cols, 1))
