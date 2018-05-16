from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import Counter

import torch
import torch.utils.data as data
from PIL import Image
import PIL
import os
import os.path
import cPickle as pickle
import numpy as np
import pandas as pd
import h5py


class SplitType:
    train = 'train'
    valid = 'validation'
    baby = 'baby'


class Dictionary(object):
    def __init__(self, path):
        print(path)
        self.word2idx = {'UNK': 0, '<eos>': 1, '<pad>': 2}
        self.idx2word = ['UNK', '<eos>', '<pad>']
        self.counter = Counter()
        self.total = len(self.idx2word)
        with open(path, 'r') as vocab:
            for i, line in enumerate(vocab.readlines()):
                word = line.decode('latin1').strip().split('\t')[0]
                self.add_word(word)
        print("Loaded dictionary with %d words" % len(self.idx2word))

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def _word2id(self, word):
        if word not in self.word2idx:
            # print "It's unkown"
            return self.word2idx['UNK']
        return self.word2idx[word]

    def words2ids(self, words):
        ids = np.asarray(map(self._word2id, words))
        return ids

    def __len__(self):
        return len(self.idx2word)

class SSENSE_Dataset(data.Dataset):
    def __init__(self, data_dir, split_name='train', orig_imsize=256, transform=None, target_transform=None, concat_category=False):
        ''' Constuctor '''
        super(SSENSE_Dataset, self).__init__()

        self.concat_category = concat_category
        self.category2idx = {
            'POCKET SQUARES & TIE BARS': 38, 'WALLETS & CARD HOLDERS': 48, 'FINE JEWELRY': 19, 'JACKETS & COATS': 5,
            'HATS': 10, 'TOPS': 0, 'SOCKS': 39, 'SHOULDER BAGS': 21, 'LOAFERS': 37, 'SHIRTS': 1, 'TIES': 8,
            'BRIEFCASES': 40, 'BELTS & SUSPENDERS': 14, 'TOTE BAGS': 27, 'TRAVEL BAGS': 47,
            'DUFFLE & TOP HANDLE BAGS': 32, 'BAG ACCESSORIES': 46, 'KEYCHAINS': 26,
            'DUFFLE BAGS': 45, 'SNEAKERS': 17, 'PANTS': 3, 'SWEATERS': 4,
            'JEWELRY': 23, 'SHORTS': 2, 'ESPADRILLES': 43, 'MESSENGER BAGS': 44,
            'EYEWEAR': 31, 'HEELS': 41, 'MONKSTRAPS': 36, 'MESSENGER BAGS & SATCHELS': 42,
            'FLATS': 33, 'BLANKETS': 22, 'POUCHES & DOCUMENT HOLDERS': 29,
            'DRESSES': 11, 'JUMPSUITS': 13, 'UNDERWEAR & LOUNGEWEAR': 25,
            'BOAT SHOES & MOCCASINS': 28, 'CLUTCHES & POUCHES': 20, 'JEANS': 6,
            'SWIMWEAR': 12, 'SUITS & BLAZERS': 7, 'LINGERIE': 16, 'GLOVES': 18, 'BOOTS': 34,
            'LACE UPS': 35, 'SCARVES': 15, 'SANDALS': 30, 'BACKPACKS': 24, 'SKIRTS': 9
        }

        self.split_name = split_name
        self.transform = transform
        self.target_transform = target_transform
        self.orig_imsize = orig_imsize
        print("Dataset Loader for Original IMSIZE: ", self.orig_imsize)

        self.data = []
        self.data_dir = data_dir
        self.data_size = 0

        split_dir = os.path.join(data_dir, self.split_name)
        print("Split Dir: %s" % split_dir)

        self.images = self.load_h5_images(split_dir)
        # self.categories = self.load_categories(split_dir)
        # self.descriptions = self.load_descriptions(split_dir)

        # if embedding_filename:
        #     self.embeddings = self.load_embedding(split_dir, embedding_filename)
        # else:
        #     self.embeddings = None

    def pad_sequence(self, seq):
        eos_id = self.dictionary.word2idx['<eos>']
        pad_id = self.dictionary.word2idx['<pad>']
        if len(seq) < self.max_desc_length:
            seq = np.concatenate([seq, [eos_id], [pad_id] * (self.max_desc_length - len(seq) - 1)])
            # seq = np.concatenate([seq, [eos_id] * (self.max_desc_length - len(seq))])
            return seq
        elif len(seq) >= self.max_desc_length:
            seq = np.concatenate([seq[:self.max_desc_length - 1], [eos_id]])
        return seq

    def get_img(self, img, imsize):
        width, height = img.size
        # load_size = int(self.imsize * 76 / 64) #might need it later
        load_size = int(imsize)
        img = img.resize((load_size, load_size), PIL.Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def load_descriptions(self, data_dir):
        filename = '/ssense_%d_%d_%s.h5' % (self.orig_imsize, self.orig_imsize, self.split_name)
        print("Loading descriptions file from %s" % filename)

        with h5py.File(data_dir + filename) as data_file:
            descriptions = np.asarray(data_file['input_description'].value)

        print('Loaded descriptions, shape: ', descriptions.shape)

        return descriptions

    def load_categories(self, data_dir):
        filename = '/ssense_%d_%d_%s.h5' % (self.orig_imsize, self.orig_imsize, self.split_name)
        print("Loading Categories file from %s" % filename)
        with h5py.File(data_dir + filename) as data_file:
            categories = np.asarray(data_file['input_category'].value)
            print('loaded Categories, shape: ', categories.shape)

        return categories

    def load_h5_images(self, data_dir):
        images_filename = '/ssense_%d_%d_%s.h5' % (self.orig_imsize, self.orig_imsize, self.split_name)
        print("Loading image file from %s" % images_filename)
        with h5py.File(data_dir + images_filename) as images_file:
            images = np.asarray(images_file['input_image'].value)
            print('loaded images, shape: ', images.shape)
            self.data_size = images.shape[0]
        return images

    def load_images(self, data_dir):
        images_filename = '/%dimages.pickle' % self.imsize
        with open(data_dir + images_filename, 'rb') as f:
            images = pickle.load(f)
            images = np.array(images)
            print('images shape: ', images.shape)

        return images

    def load_embedding(self, data_dir, embedding_filename):
        with open(os.path.join(data_dir, embedding_filename), 'rb') as f:
            embeddings = pickle.load(f)
            embeddings = np.array(embeddings).squeeze()
            print('Loaded embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        # TODO no filenames were created, but no need
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        self.examlpe_count = len(filenames)
        return filenames

    def __getitem__(self, index):
        # if self.concat_category and self.embeddings:
        #     embedding = self.embeddings[index, :]
        #     category_vector = np.zeros(len(self.category2idx), dtype=embedding.dtype)
        #     category_vector[self.category2idx[self.categories[index][0]]] = 1.
        #     embedding = np.concatenate([embedding, category_vector], axis=0)

        img = self.images[index]

        img = Image.fromarray(img.astype('uint8'), 'RGB').convert('RGB')
        # print (img)
        hr_img = self.get_img(img, self.orig_imsize)
        hr_img = hr_img.type(torch.FloatTensor)
        # print (hr_img)
        return hr_img

    def __len__(self):
        return self.data_size
