#!/usr/bin/env python
# coding: utf-8


from random import choice, random, shuffle
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nltk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split

import helpers
from helpers import idx_to_words
from model import EncoderRNN, AttnDecoderRNN


# Training loop
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, max_length):
    encoder_hidden = encoder.initHidden(device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[word2idx['<SOS>']]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            #print(decoder_input)
            #print(decoder_hidden)
            #print(encoder_outputs)
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += loss_function(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            #print(decoder_input)
            #print(decoder_hidden)
            #print(encoder_outputs)
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += loss_function(decoder_output, target_tensor[di])
            if decoder_input.item() == word2idx['<EOS>']:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length



def trainIters(encoder, decoder, n_iters, max_length, print_every=1000, plot_every=100, learning_rate=0.01):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    for iter in range(1, n_iters + 1):

        training_pair = train_set[iter - 1]
        input_tensor = training_pair[0].to(device)
        target_tensor = training_pair[1].to(device)

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, loss_function, max_length)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('(%d %d%%) %.4f' % (iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    return plot_losses


def evaluate(encoder, decoder, input_tensor):
    with torch.no_grad():
        max_length = MAX_LENGTH
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden(device)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[word2idx['<SOS>']]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == word2idx['<EOS>']:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(idx2word[str(topi.item())])

            decoder_input = topi.squeeze().detach()

        return decoded_words



def random_evaluate(evaluation_data, n=10):
    for i in range(n):
        pair = choice(evaluation_data)
        print('Instruction step', idx_to_words(pair[0], idx2word))
        print('Next step', idx_to_words(pair[1], idx2word))
        output_words = evaluate(encoder, decoder, pair[0].to(device))
        output_sentence = ' '.join(output_words)
        print('Generated instructions', output_sentence)
        print('')





# Load data
recipe_step_pairs, idx2word, word2idx, MAX_LENGTH = helpers.get_tensor_data()
n_words = len(word2idx)

#--- hyperparameters ---
N_EPOCHS = 5
LEARNING_RATE = 0.01
REPORT_EVERY = 1
HIDDEN_DIM = 256
#BATCH_SIZE = 20
#N_LAYERS = 1
teacher_forcing_ratio = 1
TRAIN_SET_SIZE = int(len(recipe_step_pairs)*0.9)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(10)


# Split into training and data set
train_set, val_set = random_split(recipe_step_pairs, [TRAIN_SET_SIZE, len(recipe_step_pairs)-TRAIN_SET_SIZE])
print(len(train_set))
print(len(val_set))

encoder = EncoderRNN(n_words, HIDDEN_DIM).to(device)
decoder = AttnDecoderRNN(HIDDEN_DIM, n_words, max_length=MAX_LENGTH).to(device)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
loss_function = nn.NLLLoss()


losses_per_epoch = []
for e in range(N_EPOCHS):
    print("---- epoch ", e)
    train_set = list(train_set)
    shuffle(train_set)
    loss = trainIters(encoder, decoder, n_iters=2, max_length=MAX_LENGTH, print_every=REPORT_EVERY)
    losses_per_epoch.append(loss)



torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'encoder_optim_state_dict': encoder_optimizer.state_dict(),
            'decoder_optim_state_dict': decoder_optimizer.state_dict(),
            'losses': losses_per_epoch
            }, './model-test.pt')


EVALUATE_N = 10
random_evaluate(evaluation_data=val_set, n=EVALUATE_N)



