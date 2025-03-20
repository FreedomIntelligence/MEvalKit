from datasets import load_dataset, Dataset
from WindaEvalKit.src.utils.utils_stategraph import *
import random
import json
from TextBase import TextBase

class TextMultiturnQA(TextBase):

    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)

