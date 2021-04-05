import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import Counter
import os
import utils
import re
import string
from argparse import Namespace

flags = Namespace(
    train_file='poetrycleaned.txt',
    seq_size=32,
    batch_size=16,
    embedding_size=64,
    lstm_size=64,
    gradients_norm=5,
    initial_words=['the', 'world'],
    predict_top_k=5,
    epochs=50,
    checkpoint_path='checkpoint',
)


def sub_with_unk(data, n=1):
    tokens = {}
    for w in data:
        if w not in tokens:
            tokens[w] = 1
        else:
            tokens[w] = tokens[w]+1
    for w in range(len(data)):
        if tokens[data[w]] <= n:
            data[w] = 'UNK'
    return data


def get_data_from_file(train_file, batch_size, seq_size,n=1):
    data = []
    with open(train_file, encoding= 'utf-8') as inp:
        lines = inp.readlines()
    for line in lines:
        line = line.strip()
        if line != '<|endoftext|>':
            line.translate(str.maketrans('', '', string.punctuation))
            data.extend(line.split(' '))
            data.append("\n")
        else:
            data.append(line)

    data = sub_with_unk(data, n)
    word_counts = Counter(data)
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    sorted_words.remove('UNK')
    sorted_words.append('UNK')
    int_to_word = {k: w for k, w in enumerate(sorted_words)}
    word_to_int = {w: k for k, w in int_to_word.items()}
    n_word = len(int_to_word)
    print(f"words: {n_word}")

    int_text = [word_to_int[w] for w in data]
    num_batches = int(len(int_text)/ (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]

    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))

    return int_to_word, word_to_int, n_word, in_text, out_text

def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]

class LanguageModel(nn.Module):
    def __init__(self, n_word, seq_size, embedding_size, hidden_size):
        super(LanguageModel, self).__init__()
        self.seq_size = seq_size
        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(n_word, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.hidden2word = nn.Linear(hidden_size, n_word)

    def forward(self, x, prev_state):
        embed = self.word_embeddings(x)
        lstm_out, hidden = self.lstm(embed, prev_state)
        output = self.hidden2word(lstm_out)

        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size)

def get_loss_and_op(model, lr= 0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return criterion, optimizer

def predict(device, model, words, n_word, word_to_int, int_to_word, top_k = 5):
    model.eval()
    for i in range(len(words)):
        if words[i] not in word_to_int:
            words[i] = 'UNK'
    state_h, state_c = model.init_hidden(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[word_to_int[w]]], dtype=torch.long).to(device)
        output, (state_h, state_c) = model(ix, (state_h, state_c))

    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()

    choice = np.random.choice(choices[0])
    while(int_to_word[choice] == 'UNK'):
        choice = np.random.choice(choices[0])
    words.append(int_to_word[choice])

    for _ in range(500):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = model(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k = top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        while (int_to_word[choice] == 'UNK'):
            choice = np.random.choice(choices[0])
        word = int_to_word[choice]
        if word == '<|endoftext|>':
            break
        words.append(word)

    print(' '.join(words))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    int_to_word, word_to_int, n_word, in_text, out_text = get_data_from_file(flags.train_file, flags.batch_size, flags.seq_size)


    model = LanguageModel(n_word, flags.seq_size, flags.embedding_size, flags.lstm_size)

    model = model.to(device)

    criterion, optimizer = get_loss_and_op(model, 0.01)

    iteration = 0

    for e in range(flags.epochs):
        batches = get_batches(in_text, out_text, flags.batch_size, flags.seq_size)
        state_h, state_c = model.init_hidden(flags.batch_size)
        state_h = state_h.to(device)
        state_c = state_c.to(device)

        for x, y in batches:
            iteration += 1
            model.train()
            optimizer.zero_grad()
            x = torch.tensor(x, dtype=torch.long).to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)

            output, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(output.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss_value = loss.item()

            loss.backward()

            _ = torch.nn.utils.clip_grad_norm_(
                model.parameters(), flags.gradients_norm)

            optimizer.step()

            if iteration % 100 == 0:
                print('Epoch: {}/{}'.format(e, 50),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value))

            if iteration % 1000 == 0:
                predict(device, model, flags.initial_words, n_word,
                        word_to_int, int_to_word, top_k=5)
                torch.save(model.state_dict(),'checkpoint_pt/model-{}.pth'.format(iteration))


if __name__ == '__main__':
    main()

























