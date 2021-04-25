import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import Counter
import os

import re
import string
import pronouncing
from argparse import Namespace

flags = Namespace(
    train_file='formattedcleanedrhymepoetry.txt',
    test_file='fcrtestset.txt',
    seq_size=8,
    batch_size=32,
    layers=2,
    embedding_size=128,
    lstm_size=128,
    gradients_norm=5,
    initial_words=[['<s>', 'we', 'tried'], ['<s>', 'ours', 'were'], ['<s>', 'she', 'was', 'so', 'strong']],
    predict_top_k=4,
    epochs=500,
    bidir=False,
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


def get_data_from_file_for_testing(test_file, batch_size, seq_size, word_to_int):
    data = []
    with open(test_file, encoding= 'utf-8') as inp:
        lines = inp.readlines()
    for line in lines:
        line = line.strip().split(' ')
        if line[-1] == '</s>':
            data.extend(line)
        else:
            data.extend(line)
            data.append('\n')
    int_text = []
    for w in data:
        if w not in word_to_int:
            int_text.append(word_to_int['UNK'])
        else:
            int_text.append(word_to_int[w])
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]

    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))

    return in_text, out_text


def get_data_from_file(train_file, batch_size, seq_size,n=1):
    data = []
    with open(train_file, encoding= 'utf-8') as inp:
        lines = inp.readlines()
    for line in lines:
        line = line.strip().split(' ')
        if line[-1] == '</s>':
            data.extend(line)
        else:
            data.extend(line)
            data.append('\n')

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
    def __init__(self, n_word, seq_size, embedding_size, hidden_size, num_layers, bidir):
        super(LanguageModel, self).__init__()
        self.seq_size = seq_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidir = bidir
        self.word_embeddings = nn.Embedding(n_word, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidir, dropout=.3)
        self.dropout = nn.Dropout(.3)
        self.direction = 1
        if bidir == True:
            self.direction = 2
        self.hidden2word = nn.Linear(hidden_size * self.direction, n_word)

    def forward(self, x, prev_state):
        embed = self.word_embeddings(x)
        lstm_out, hidden = self.lstm(embed, prev_state)
        output = self.dropout(lstm_out)
        output = self.hidden2word(output)
        #output = F.log_softmax(output, dim=1)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers * self.direction, batch_size, self.hidden_size), torch.zeros(self.num_layers * self.direction, batch_size, self.hidden_size)

def get_loss_and_op(model, lr= 0.1):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return criterion, optimizer

def evaluate(device, model, words, n_word, word_to_int, int_to_word):
    model.eval()

def generatePoem(device, model, words, n_words, word_to_int, int_to_word, lines):
    # save the last word from the previous line
    prev_word = words[-1]
    syllables = 0
    # count the number of syllables in the line
    for w in words[1:]:
        syllables += pronouncing.syllable_count(pronouncing.phones_for_word(w)[0])
    # count the number of syllables in the line
    states = []
    line = []
    states.append()
    for l in range(1,lines):
        states = []

    # try to find an output with the similar number of syllables and try to rhyme with the previous line

    words.append('\n')
    for i in range(len(words)):
        if (words[i] not in word_to_int):
            words[i] = 'UNK'

    for l in range(1, lines):
        states = []



def predictPoem(device, model, words, n_word, word_to_int, int_to_word, length, top_k = 5):
    model.eval()
    model.zero_grad()
    for i in range(len(words)):
        if words[i] not in word_to_int:
            words[i] = 'UNK'
    state_h, state_c = model.init_hidden(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    temp = []
    for w in words:
        ix = torch.tensor([[word_to_int[w]]], dtype=torch.long).to(device)
        output, (state_h, state_c) = model(ix, (state_h, state_c))
        temp.append(w)

    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    options = []
    for c in choices:
        options.append(int_to_word[c])
    return options


def predict(device, model, words, n_word, word_to_int, int_to_word, length, top_k = 5):
    model.eval()
    model.zero_grad()
    for i in range(len(words)):
        if words[i] not in word_to_int:
            words[i] = 'UNK'
    state_h, state_c = model.init_hidden(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    temp = []
    for w in words:
        ix = torch.tensor([[word_to_int[w]]], dtype=torch.long).to(device)
        output, (state_h, state_c) = model(ix, (state_h, state_c))
        temp.append(w)

    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()

    choice = np.random.choice(choices[0])
    while(int_to_word[choice] == 'UNK' and top_k != 1):
        choice = np.random.choice(choices[0])
    temp.append(int_to_word[choice])
    last_rhyme = -1;
    lines = 0
    for _ in range(length-1):
        ix = torch.tensor([[choice]], dtype=torch.long).to(device)
        output, (state_h, state_c) = model(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        while ((int_to_word[choice] == 'UNK' and top_k != 1) or int_to_word[choice] == temp[-1] or(int_to_word[choice] == '</s>' and lines < 4) or int_to_word[choice] == '<s>'):
            choice = np.random.choice(choices[0])
        if int_to_word[choice] == "\n":
            lines += 1
        if int_to_word[choice] == '</s>':
            break
        word = int_to_word[choice]
        temp.append(word)
    # poem = ''
    # for word in temp:
    #     if word == '</s>':
    #         break
    #     elif word == '<s>':
    #         continue
    #     else:
    #         poem = poem + ' ' + word
    # poem = poem.strip()
    print(' '.join(temp[1:]))



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    int_to_word, word_to_int, n_word, in_text, out_text = get_data_from_file(flags.train_file, flags.batch_size, flags.seq_size)

    vin_text, vout_text = get_data_from_file_for_testing(flags.test_file, flags.batch_size, flags.seq_size,word_to_int)

    model = LanguageModel(n_word, flags.seq_size, flags.embedding_size, flags.lstm_size,flags.layers, flags.bidir)

    model = model.to(device)

    criterion, optimizer = get_loss_and_op(model, 0.001)

    iteration = 0

    for e in range(flags.epochs):
        batches = get_batches(in_text, out_text, flags.batch_size, flags.seq_size)
        state_h, state_c = model.init_hidden(flags.batch_size)
        state_h = state_h.to(device)
        state_c = state_c.to(device)
        #loss_avg =
        for x, y in batches:
            iteration += 1
            model.train()
            optimizer.zero_grad()
            # model.zero_grad()

            x = torch.tensor(x, dtype=torch.long).to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)

            output, (state_h, state_c) = model(x, (state_h, state_c))
            # print(f"output shape:{output.size()}, target shape{y.size()}")
            loss = criterion(output.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()

            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), flags.gradients_norm)

            optimizer.step()

        torch.save(model.state_dict(), 'checkpoint_pt/model-{}.pth'.format(iteration))
        print('Epoch: {}/{}'.format(e, flags.epochs),
              'Iteration: {}'.format(iteration),
              'Loss: {}'.format(loss.item()), 'Perplexity: {}'.format(torch.exp(loss)))

        validbatches = get_batches(vin_text, vout_text, flags.batch_size, flags.seq_size)
        state_h, state_c = model.init_hidden(flags.batch_size)
        state_h = state_h.to(device)
        state_c = state_c.to(device)

        for x, y in validbatches:
            iteration += 1
            model.eval()
            # optimizer.zero_grad()
            model.zero_grad()

            x = torch.tensor(x, dtype=torch.long).to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)

            output, (state_h, state_c) = model(x, (state_h, state_c))
            # print(f"output shape:{output.size()}, target shape{y.size()}")
            loss = criterion(output.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()
            perplexity = torch.exp(loss)
            loss_value = loss.item()

        print('Epoch: {}/{}'.format(e, flags.epochs),
              'Test Loss: {}'.format(loss.item()), 'Test Perplexity: {}'.format(torch.exp(loss)))
        for prompt in flags.initial_words:
            predict(device, model, prompt, n_word, word_to_int, int_to_word,100, top_k=4)
            print('\n')
    torch.save(model.state_dict(), 'checkpoint_pt/model-{}.pth'.format(iteration))


if __name__ == '__main__':
    main()

























