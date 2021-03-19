import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.autograd import Variable
import numpy as np
import utils
import time, math



torch.manual_seed(42)

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)



class LanguageModel(nn.Module):

    def __init__(self, input_size, hidden_dim,  output_size,numlay):
        super(LanguageModel, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.numlay = numlay

        self.word_embeddings = nn.Embedding(input_size, hidden_dim)
        #reason why its currently gru and not LSTM is because gru is simpler
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=numlay)
        self.hidden2word = nn.Linear(hidden_dim, output_size)
    # given one word at a time

    def forward(self, input, hidden):
        input = self.word_embeddings(input.view(1,-1))
        output, hidden = self.gru(input.view(1,1,-1), hidden)
        output = self.hidden2word(output.view(1, -1))
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.numlay, 1, self.hidden_dim)


TRAINING_FILE = "data/poetrycleaned.txt"
data = utils.readData(TRAINING_FILE)
training_data = data[:-len(data)//16]
word_to_ix = {}
ix_to_word = {}


for sent in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
            ix_to_word[word_to_ix[word]] = word

def getRandomPoem():
    index = random.randint(0, len(training_data))
    return training_data[index]

def getTrainingSet(poem):
    input = prepare_sequence(poem[:-1], word_to_ix)
    target = prepare_sequence(poem[1:], word_to_ix)
    length = len(poem)
    return input, target, length

def evaluate(starting_str = 'The world', predict_len = 100, temperature = 0.8):
    hidden = model.initHidden()
    prepared_sentence = starting_str.lower().split()
    input = prepare_sequence(prepared_sentence, word_to_ix)
    predicted = starting_str

    for w in range(len(prepared_sentence) - 1):
        _, hidden = model(input[w], hidden)
    inp = input[-1]

    for p in range(predict_len):
        output, hidden = model(inp, hidden)

        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist,1)[0]

        predicted_word = ix_to_word[top_i]
        predicted += ' ' + predicted_word
        inp = word_to_ix(predicted_word)
    return predicted

def train(input, target,poem_length):
    hidden = model.initHidden()
    model.zero_grad()
    loss = 0
    for w in range(poem_length):
        output, hidden = model(input[w], hidden)
        print(f'{output.shape}')
        print(f'{target[w]}')
        loss += loss_function(output, target[w])
    loss.backward()
    optimizer.step()

    return loss.data[0] / poem_length

EMBEDDING_DIM = 32
HIDDEN_DIM = 32
LAYERS = 1


model = LanguageModel(len(word_to_ix), HIDDEN_DIM, len(word_to_ix), LAYERS)

loading = False
modelpath = 'poetryLM.pt'
#testfile
#outputpath


if (loading == False):
    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_avg = 0
    #Training Loop
    for epoch in range(10):
        print(f"Starting epoch {epoch}...")
        random.shuffle(training_data)
        for poem in training_data:
            loss = train(*getTrainingSet(poem))
            loss_avg += loss

        print(f"epoch {epoch}/10, loss {loss_avg/ len(training_data)}")
        print(evaluate('The', 100), '\n')

        torch.save(model, modelpath)
else:
    model = torch.load(modelpath)
# model.eval()
#
# start = []
# start.append('<s>')
# max_length = 200
# with torch.no_grad():
#     input = prepare_sequence(start, word_to_ix)
#     hidden = model.initHidden()
#     output_poem = ''
#     for i in range(max_length):
#         output, hidden = model(input[0], )







