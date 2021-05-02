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
    testing=True,
    train_file='formattedcleanedrhymepoetry.txt',
    test_file='fcrtestset.txt',
    save_models=True,
    seq_size=32,
    batch_size=32,
    layers=2,
    embedding_size=256,
    lstm_size=256,
    gradients_norm=5,
    initial_words=[['<s>', 'we', 'tried'], ['<s>', 'ours', 'were'], ['<s>', 'she', 'was', 'so', 'strong']],
    predict_top_k=4,
    epochs=500,
    bidir=False,
    syllables=14,
    beam_size=10,
    checkpoint_path='checkpoint',
    test_model_file='checkpoint_pt/model-4636.pth'
)

# substitutes tokens in the dataset that only appear n times with UNK
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

#used to load the data from the testing file
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

# used to load the data from the training file
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

#creates batches using the given data
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
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidir, dropout=.1)
        self.dropout = nn.Dropout(.1)
        self.direction = 1
        if bidir == True:
            self.direction = 2
        self.hidden2word = nn.Linear(hidden_size * self.direction, n_word)

    def forward(self, x, prev_state):
        embed = self.word_embeddings(x)
        lstm_out, hidden = self.lstm(embed, prev_state)
        output = self.dropout(lstm_out)
        output = output.reshape(-1, self.hidden_size)
        output = self.hidden2word(output)
        #output = F.log_softmax(output, dim=1)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers * self.direction, batch_size, self.hidden_size), torch.zeros(self.num_layers * self.direction, batch_size, self.hidden_size)

def get_loss_and_op(model, lr= 0.1):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return criterion, optimizer



# uses beamsearch to try and generate poetry
def generateRhymingPoem(device, model, words, word_to_int, int_to_word, meter, top_k=10, beam = 5):
    # create starting state from words
    symbols = ['<s>', '</s>', '\n', 'UNK']
    states = []
    found = False
    start = [0, -1, words.copy(), 1]
    #first count the number of syllables in the initial state
    for w in words:
        # if w not in the symbols list get the numbe
        if w not in symbols:
            phones_list = pronouncing.phones_for_word(w)
            start[0] += pronouncing.syllable_count(phones_list[0])
    states.append(start)
    # create all the possible first lines using the initial words
    first_lines = []
    i = 0
    while not found:
        #print(i)
        i += 1
        newstates = []
        for s in states:
            # get the top_k possible next words using the given set of words
            # print(s)
            new_words = getPossibleNextWords(device, model, s[2], word_to_int, int_to_word, top_k)
            #print(new_words)
            for nw in new_words:
                # if the new word isn't a symbol and it is not a repeat of the previous word in the state
                if nw[0] not in symbols and nw[0] != s[2][-1] and nw[0] != s[2][-2]:
                    # check to see if the word exists in the pronunciation dictionary
                    phones_list = pronouncing.phones_for_word(nw[0])

                    if len(phones_list) > 0 and pronouncing.syllable_count(phones_list[0]) + s[0] <= meter:
                        # if it does exist and the number of syllables plus the syllables in the previous words don't go over the meter
                        # add it to the new states
                        rate = 1
                        if(pronouncing.syllable_count(phones_list[0]) + s[0] == meter):
                            rate = 2
                        ts = s[2].copy()
                        ts.append(nw[0])
                        newstates.append([s[0] + pronouncing.syllable_count(phones_list[0]), s[1], ts, s[3] * nw[1]])
                elif nw[0] == '\n' and s[0] == meter:
                    # else if the new word is a newline and the correct meter has been met
                    # store state with 0 as the number of syllables, the last word in the sentence, the new sentence, and its probability
                    # put it in the first lines list
                    ts = s[2].copy()
                    ts.append('\n')
                    first_lines.append([0, s[2][-1], ts, s[3] * nw[1]])
        if len(newstates) == 0:
            # if there are no more new states leave the loop
            break
        else:
            # if there are more new states then sort them by probability
            newstates.sort(reverse=True, key=lambda state: state[3])
            states = newstates.copy()
            states = states[:beam]
    finished_poems = []
    if len(first_lines) == 0:
        print("no valid first lines found")

    else:
        print("First Lines:")
        for l in first_lines:
            print(f"{' '.join(l[2][1:])}")
        #perform beam search again using the first lines as the states
        # print(len(first_lines))
        states = first_lines
        while not found:
            newstates = []
            for s in states:
                new_words = getPossibleNextWords(device, model, s[2], word_to_int, int_to_word, top_k)
                for nw in new_words:
                    if nw[0] not in symbols and nw[0] != s[2][-1]:
                        phones_list = pronouncing.phones_for_word(nw[0])
                        if len(phones_list) > 0 and pronouncing.syllable_count(phones_list[0]) + s[0] <= meter:
                            if pronouncing.syllable_count(phones_list[0]) + s[0] == meter:

                                if nw[0] in pronouncing.rhymes(s[1]):
                                    # if the meter is met and the new word rhymes with the last word from the previous line
                                    # add it to newstates with an increased probability
                                    ts = s[2].copy()
                                    ts.append(nw[0])
                                    newstates.append([s[0] + pronouncing.syllable_count(phones_list[0]), s[1], ts, s[3] * nw[1] * 10])
                            else:
                                ts = s[2].copy()
                                ts.append(nw[0])
                                newstates.append([s[0] + pronouncing.syllable_count(phones_list[0]), s[1], ts, s[3] * nw[1]])
                        elif nw[0] == '</s>' and s[0] == meter:
                            finished_poems.append([s[2], s[3] * nw[1]])
            if len(newstates) == 0:
                break
            else:
                newstates.sort(reverse=True, key=lambda state: state[3])
                states = newstates.copy()
                states = states[:beam]
        if len(finished_poems) == 0:
            print("No possible poems found")
        else:
            print(f"Poems made using: {' '.join(words)}")
            finished_poems.sort(reverse=True, key=lambda state: state[1])
            for fp in finished_poems[:beam]:
                print(f"{' '.join(fp[0][1:])}\n")


def getPossibleNextWords(device, model, words, word_to_int, int_to_word, top_k = 5):
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
    p = F.softmax(output, dim=1).data
    p = p.cpu()
    p = p.numpy()
    p = p.reshape(p.shape[1],)
    probabilities = np.sort(p)[-top_k:][::-1]
    top_n_idx = p.argsort()[-top_k:][::-1]
    choices = top_n_idx.tolist()
    #_, top_ix = torch.topk(output[0], k=top_k)
    #choices = top_ix.tolist()
    options = []
    for c in range(len(choices)):
        options.append((int_to_word[c], probabilities[c]))
    return options


def predict(device, model, words, n_word, word_to_int, int_to_word, length, top_k = 5):
    model.eval()
    # model.zero_grad()
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
    p = F.softmax(output, dim=1).data
    p = p.cpu()
    p = p.numpy()
    p = p.reshape(p.shape[1], )
    probabilities = np.sort(p)[-top_k:][::-1]
    #print(probabilities)
    top_n_idx = p.argsort()[-top_k:][::-1]
    #print(top_n_idx)
    choices = np.asarray(top_n_idx)
    #choice = np.random.choice(choices)
    #print(top_n_idx)
    #choices = top_n_idx.tolist()
    ##rand_index = top_n
    #_, top_ix = torch.topk(output[0], k=top_k)
    #choices = top_ix.tolist()

    choice = np.random.choice(choices)
    while((int_to_word[choice] == 'UNK' and top_k != 1) or int_to_word[choice] == '<s>' or int_to_word[choice] == '</s>'):
        choice = np.random.choice(choices)
    temp.append(int_to_word[choice])
    last_rhyme = -1
    lines = 0
    for _ in range(length-1):
        ix = torch.tensor([[choice]], dtype=torch.long).to(device)
        output, (state_h, state_c) = model(ix, (state_h, state_c))
        p = F.softmax(output, dim=1).data

        p = p.cpu()
        p = p.numpy()
        p = p.reshape(p.shape[1], )
        # print(p.argsort()[-top_k:])

        top_n_idx = p.argsort()[-top_k:][::-1]
        #print(top_n_idx)
        choices = np.asarray(top_n_idx)
        # _, top_ix = torch.topk(output[0], k=top_k)
        # choices = top_ix.tolist()
        choice = np.random.choice(choices)
        while ((int_to_word[choice] == 'UNK' and top_k != 1) or int_to_word[choice] == temp[-1] or(int_to_word[choice] == '</s>' and lines < 3) or int_to_word[choice] == '<s>'):
            choice = np.random.choice(choices)
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
    if flags.testing:
        model = torch.load(flags.test_model_file)

    model = model.to(device)

    if not flags.testing:
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
                loss = criterion(output, y.view(-1))

                state_h = state_h.detach()
                state_c = state_c.detach()

                loss.backward()

                _ = torch.nn.utils.clip_grad_norm_(model.parameters(), flags.gradients_norm)

                optimizer.step()
            if flags.save_models:
                torch.save(model, 'checkpoint_pt/model-{}.pth'.format(iteration))
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
                loss = criterion(output, y.view(-1))

                state_h = state_h.detach()
                state_c = state_c.detach()
                perplexity = torch.exp(loss)
                loss_value = loss.item()

            print('Epoch: {}/{}'.format(e, flags.epochs),
                  'Test Loss: {}'.format(loss.item()), 'Test Perplexity: {}'.format(torch.exp(loss)))
            for prompt in flags.initial_words:
                predict(device, model, prompt, n_word, word_to_int, int_to_word,100, top_k=5)
                print('\n')
                #generateRhymingPoem(device, model, prompt, word_to_int, int_to_word, 7, top_k=10, beam=50)
                #print('\n')
            # torch.save(model.state_dict(), 'checkpoint_pt/model-{}.pth'.format(iteration))
    else:
        for prompt in flags.initial_words:
            predict(device, model, prompt, n_word, word_to_int, int_to_word, 100, top_k=5)
            print('\n')
            #generateRhymingPoem(device, model, prompt, word_to_int, int_to_word,flags.syllables, top_k=10, beam=flags.beam_size)
            #print('\n')

if __name__ == '__main__':
    main()

























