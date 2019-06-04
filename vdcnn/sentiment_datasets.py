import os
import abc
from abc import ABCMeta
import torch

import pandas as pd
import numpy as np
import torch.utils.data as data

class DataPrefetcher():
    def __init__(self, loader, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None

    def __len__(self):
        return len(self.loader)

    def _prefetch(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self._prefetch()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self._prefetch()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break

class Vocab:
    def __init__(self, tokens=None, offset=0, unknown=None):
        self.mapping = {}
        self.reverse_mapping = {}
        self.offset = offset
        self.unknown = unknown
        for token in tokens:
            self.add_token(token)

    def __len__(self):
        return len(self.mapping) + self.offset

    def __call__(self, doc):
        return self.map_sequence(doc)

    def add_token(self, token):
        if token not in self.mapping:
            index = len(self)
            self.mapping[token] = index
            self.reverse_mapping[index] = token

    def __repr__(self):
        fmt_str = self.__class__.__name__
        fmt_str += "(vocab={0}, offset={1}, unknown={2})".format(
            self.__len__(), self.offset, self.unknown)
        return fmt_str

    def map(self, token, unknown=None):
        if token in self.mapping:
            return self.mapping[token]
        else:
            return unknown if unknown is not None else self.unknown

    def map_sequence(self, tokens, unknown=None):
        return np.array([self.map(token, unknown=unknown) for token in tokens])

    def reverse_map(self, index, unknown=None):
        if index in self.reverse_mapping:
            return self.reverse_mapping[index]
        else:
            return unknown if unknown is not None else self.unknown

    def reverse_map_sequence(self, indices, unknown):
        return [self.reverse_map(index, unknown=unknown) for index in indices]


class PadOrTruncate:
    def __init__(self, max_length, fill=0):
        self.max_length = max_length
        self.fill = fill

    def __call__(self, doc):
        current = len(doc)
        trimmed = doc[:self.max_length]
        padding = [self.fill] * (self.max_length - current)
        return np.concatenate([trimmed, padding])

    def __repr__(self):
        fmt_str = self.__class__.__name__
        fmt_str += "(max_length={0}, fill={1})".format(
            self.max_length, self.fill)
        return fmt_str

class TextClassification(data.Dataset, metaclass=ABCMeta):

    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            self.train_data, self.train_labels = self.load_train_data()
            assert self.train_labels.dtype == np.int64
        else:
            self.test_data, self.test_labels = self.load_test_data()
            assert self.test_labels.dtype == np.int64

        self._classes = None

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (doc, target) where target is index of the target class.
        """
        if self.train:
            doc, target = self.train_data[index], self.train_labels[index]
        else:
            doc, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            doc = self.transform(doc)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return doc, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))  # noqa: E501
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))  # noqa: E501
        return fmt_str

    @abc.abstractmethod
    def load_train_data(self):
        pass

    @abc.abstractmethod
    def load_test_data(self):
        pass

    @property
    def classes(self):
        if self._classes is None:
            self._classes = len(set(self.train_labels)) if self.train else len(set(self.test_labels))  # noqa: E501
        return self._classes


class XiangZhangDataset(TextClassification):
    def load_train_data(self):
        assert self.dirname in self.root
        return self.load_data(os.path.join(self.root, 'train.csv'))

    def load_test_data(self):
        assert self.dirname in self.root
        return self.load_data(os.path.join(self.root, 'test.csv'))

    def load_data(self, path):
        df = pd.read_csv(path, header=None, keep_default_na=False,
                         names=self.columns)
        labels = (df[self.columns[0]] - df[self.columns[0]].min()).values
        if len(self.columns) > 2:
            data = (df[self.columns[1]]
                    .str
                    .cat([df[col] for col in self.columns[2:]], sep=" ")
                    .values)
        else:
            data = df[self.columns[1]].values
        return data, labels

    @property
    @abc.abstractmethod
    def dirname(self):
        pass

    @property
    @abc.abstractmethod
    def columns(self):
        pass


class AmazonReviewPolarity(XiangZhangDataset):
    dirname = "amazon_review_polarity_csv"
    columns = ['rating', 'subject', 'body']


class AmazonReviewFull(AmazonReviewPolarity):
    dirname = "amazon_review_full_csv"


class AGNews(XiangZhangDataset):
    dirname = "ag_news_csv"
    columns = ['class_index', 'title', 'description']


class DBPedia(XiangZhangDataset):
    dirname = "dbpedia_csv"
    columns = ['class_index', 'title', 'content']


class SogouNews(XiangZhangDataset):
    dirname = "sogou_news_csv"
    columns = ['class_index', 'title', 'content']


class YahooAnswers(XiangZhangDataset):
    dirname = "yahoo_answers_csv"
    columns = ['class_index', 'question_title', 'question_content',
               'best_answer']


class YelpReviewFull(XiangZhangDataset):
    dirname = "yelp_review_full_csv"
    columns = ['rating', 'review']


class YelpReviewPolarity(YelpReviewFull):
    dirname = "yelp_review_polarity_csv"

DATASETS = {
    'amazon_review_full': AmazonReviewFull,
    'amazon_review_polarity': AmazonReviewPolarity,
    'ag_news': AGNews,
    'dbpedia': DBPedia,
    'sogou_news': SogouNews,
    'yahoo_answers': YahooAnswers,
    'yelp_review_full': YelpReviewFull,
    'yelp_review_polarity': YelpReviewPolarity,
}
