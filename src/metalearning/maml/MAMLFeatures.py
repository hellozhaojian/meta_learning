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
@file: MAMLFeatures.py
@time: 2020/1/28 1:24 下午
"""

import sys, os
from metalearning.maml.config import MAMLConfig
from metalearning.maml.domain import DataType
import random
from typing import Tuple, List

"""
meta learning 和一般的生成train 和  Test 不一样，
"""


class MAMLFeatures(object):

    def __init__(self, config: MAMLConfig):
        self.config = config
        self.train_ratio: float = self.config.data_ratios[DataType.TRAIN.value]
        self.dev_ratio: float = self.config.data_ratios[DataType.DEV.value]
        self.test_ratio: float = self.config.data_ratios[DataType.TEST.value]
        has_omniglot = os.path.isdir(self.config.omniglot_path)
        has_min_imagenet = os.path.isdir(self.config.mini_image_path)
        assert has_omniglot is True
        assert has_min_imagenet is True

    def generate_omniglot_dataset(self) -> Tuple:
        data_dir = self.config.omniglot_path
        character_folders = [os.path.join(data_dir, family, character)
                             for family in os.listdir(data_dir)
                             if os.path.isdir(os.path.join(data_dir, family))
                             for character in os.listdir(os.path.join(data_dir, family))]
        print(character_folders[:10])
        data_length = len(character_folders)
        random.seed(self.config.random_seed)
        random.shuffle(character_folders)
        train_count = int(self.train_ratio * data_length)
        valid_count = int(self.dev_ratio * data_length)
        test_count = int(data_length - train_count - valid_count)
        print(train_count, valid_count, test_count)
        train_dataset = character_folders[:train_count]
        valid_dataset = character_folders[train_count:train_count + valid_count]
        test_dataset = character_folders[train_count + valid_count:train_count + valid_count + test_count]
        return train_dataset, valid_dataset, test_dataset

    def _generate_mini_imagenet_dataset(self, data_type: str) -> List:
        return [os.path.join(self.config.mini_image_path, data_type, family)
                for family in os.listdir(os.path.join(self.config.mini_image_path, data_type))
                if os.path.isdir(os.path.join(self.config.mini_image_path, data_type, family))
                ]

    def generate_mini_imagenet_dataset(self) -> Tuple:
        train_classes = self._generate_mini_imagenet_dataset('train')
        dev_classes = self._generate_mini_imagenet_dataset('val')
        test_classes = self._generate_mini_imagenet_dataset("test")
        print(train_classes[:10])
        return train_classes, dev_classes, test_classes




if __name__ == "__main__":
    config = MAMLConfig()
    feature_processor = MAMLFeatures(config)
    feature_processor.generate_omniglot_dataset()
    feature_processor.generate_mini_imagenet_dataset()
