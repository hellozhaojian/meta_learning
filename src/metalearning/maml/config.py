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
@file: config.py
@time: 2020/1/28 11:00 上午
"""

import sys, os
from pydantic import BaseModel
import yaml
from pathlib import Path
from metalearning.maml.domain import DataType


class MAMLConfig(BaseModel):

    learning_rate: float = 0.1
    data_ratios = {DataType.TRAIN.value: 0.9, DataType.DEV.value: 0.05, DataType.TEST.value: 0.05}

    mini_image_path: str = "/Users/alchemy_taotaox/nltk_data/mini-imagenet"
    omniglot_path: str = "/Users/alchemy_taotaox/nltk_data/omniglot_resized"
    random_seed: int = 298998999

    @staticmethod
    def load(filename):
        with open(filename, mode="r") as file_h:
            tmp = yaml.safe_load(file_h)
            return MAMLConfig(**tmp)

    def dump(self, filename):
        with open(filename, mode="w") as file_h:
            file_h.write(yaml.dump(self.dict()))


if __name__ == "__main__":
    print(DataType.TEST.value)
    obj = MAMLConfig()
    obj.dump("test.yml")
    print(obj.learning_rate)
    x = MAMLConfig.load("test.yml")
    print(x.learning_rate)