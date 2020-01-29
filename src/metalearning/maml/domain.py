# encoding: utf-8
# Copyright 2019 The DeepNlp Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
"""
@file: domain.py
@time: 2020/1/28 4:21 下午
"""
import enum
import random
import sys, os


class DataType(enum.Enum):

    TRAIN = "train"
    TEST = "test"
    DEV = "dev"


class Task(object):
    """
    An abstract class for defining a single few-shot task
    """

    def __init__(self, image_folders, num_class, support_num, query_num):
        """
        train_* are support set
        test_* are a query set
        meta_* are for meta update in meta_learner
        :param image_folders:
        :param num_class:
        :param support_num:
        :param query_num:
        """
        self.image_folders = image_folders
        self.num_class = num_class
        self.support_num = support_num
        self.query_num = query_num

        class_folders = random.sample(self.image_folders, self.num_class)
        labels = list(range(len(class_folders)))
        folders_2_labels = dict(zip(class_folders, labels))

        self.train_roots = []
        self.train_labels = []
        self.test_roots = []
        self.test_labels = []
        for c in class_folders:
            all_image_files = [os.path.join(c, x) for x in os.listdir(c)]
            samples = random.shuffle(all_image_files)
            self.train_roots += samples[:self.support_num]
            self.train_labels += [ folders_2_labels[c] for _ in samples[:self.support_num]]
            self.test_roots += samples[self.support_num:self.support_num + self.query_num]
            self.test_labels += [ folders_2_labels[c] for _ in samples[self.support_num:self.support_num+self.query_num]]

        self.meta_roots = []
        self.meta_labels = []
        for c in class_folders:
            all_image_files = [os.path.join(c,x) for x in os.listdir(c)]
            samples= random.shuffle(all_image_files)
            self.meta_roots += samples[:self.support_num]
            self.meta_labels += [ folders_2_labels[c] for _ in samples[:self.support_num]]


class OmniglotTask(Task):

    def __init__(self, *args, **kwargs):
        super(OmniglotTask, self).__init__(*args, **kwargs)


class ImagenetTask(Task):

    def __init__(self, *args, **kwargs):
        super(ImagenetTask, self).__init__(*args, **kwargs)


# class



