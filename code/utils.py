from email.parser import Parser
import imp
import json
import os
from typing import OrderedDict
from tqdm import tqdm
import sys
import logging

import random
logger = logging.getLogger(__name__)

import torch
import numpy as np
import orjson

def load_vocab(vocab_filename):
    with open(vocab_filename, "r") as f:
        lines = f.readlines()
    vocab = [x.strip() for x in lines]
    return vocab

def load_jsonl(filename):
    try:
        data = []
        with open(filename, "r") as f:
            for line in f.readlines():
                data.append(orjson.loads(line))
        return data
    except Exception as e:
        print(f"error: cannot open the {filename}")
        print(e)
        exit(-1)

def load_json(filename):
    try:
        with open(filename,"r") as f:
            data =  json.load(f)
        return data
    except:
        print(f"error: cannot open the {filename}")
        exit(-1)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)


