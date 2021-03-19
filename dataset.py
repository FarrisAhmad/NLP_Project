import torch
import pandas as pd
from collections import Counter

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
    ):
        self.args = args
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        with open('data/poetrycleaned.txt', encoding='utf-8') as inp:
            lines = inp.readlines()
        first = True
        data = []
        split_count = 4
        final_data = []
        for line in lines:
            line = line.strip()
            if(line == '<|endoftext|>'):
                data[-1].append('<eos>')
                first = True
                continue
            elif first:
                data.append([])
                data[-1].append('<sos>')
                first = False
            data[-1].extend(line.split())
            data[-1].append('<nl>')
        count = len(data)
        data = data[:count//4]
        for p in data:
            final_data.extend(p)
        return final_data

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.args.sequence_length]),
            torch.tensor(self.words_indexes[index+1:index+self.args.sequence_length+1]),
        )
