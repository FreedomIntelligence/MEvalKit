from datasets import load_dataset, Dataset, get_dataset_config_names
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
import torch
import re
import os
import json
from utils import load_dataset_info

class ImageMCQDataset:
    def __init__():
        pass

    def load_model_and_tokenizer(self, model_name):
        self.model = AutoModelForVision2Seq.from_pretrained(model_name)
        
