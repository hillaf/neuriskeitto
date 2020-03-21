#!/usr/bin/env python
# coding: utf-8

import json
from random import choice, random, shuffle
import random
import sys

import numpy as np
import torch
import torch.nn as nn



# Helper functions
def idx_to_words(indices, idx2word):
    return " ".join([idx2word[str(i.item())] for i in indices])


def get_tensor_data():
    with open('project_data/project_train_data_ingr.json') as json_file:
        all_recs_ingr = json.load(json_file)
    with open('project_data/project_train_data_instr.json') as json_file:
        all_recs_instr = json.load(json_file)
    with open('project_data/project_train_data_word2idx.json') as json_file:
        word2idx = json.load(json_file)
    with open('project_data/project_train_data_idx2word.json') as json_file:
        idx2word = json.load(json_file)

    # Preprocess: to idx representation
    input_ingr = []
    for recipe in all_recs_ingr:
        input_ingr.append([word2idx[w] for w in recipe])
    print(input_ingr[0])

    input_instr = []
    instr_tensors = []
    for recipe in all_recs_instr:
        steps = []
        step_tensors = []
        for step in recipe:
            idx_list = [word2idx[w] for w in step]
            steps.append(idx_list)
            step_tensors.append(torch.tensor(idx_list).view(-1, 1))
        input_instr.append(steps)
        instr_tensors.append(step_tensors)
    #print(input_instr[0])
    #print(instr_tensors[0])



    # Create tensors
    input_ingr_tensors = [torch.tensor(rec).view(-1, 1) for rec in input_ingr]
    #input_instr_tensors = torch.tensor()
    #for rec in input_instr:
    #    input_instr_tensors.append([torch.tensor(step).view(-1, 1) for step in rec])
    recipe_pairs = [(input_ingr_tensors[i], instr_tensors[i]) for i, rec in enumerate(input_ingr_tensors)]

    #print(len(input_ingr_tensors))
    #print(input_ingr_tensors[0])

    # Sanity check
    recipe = choice(recipe_pairs)


    #print(" ".join([idx2word[i.item()] for i in recipe[0]]))
    #print(" ".join([idx2word[i.item()] for i in recipe[1]]))


    sum_len_instr = 0
    count_short = 0
    count_short_instr = 0
    MAX_LENGTH = 0
    total_steps = 0
    longest_instr = []
    rm_indices = set()

    for i, rec in enumerate(instr_tensors):
        if len(rec) < 2:
            rm_indices.add(i)
        for step in rec:
            total_steps = total_steps + 1
            sum_len_instr = sum_len_instr + len(step)
            if len(step) > 40:
                rm_indices.add(i)
                count_short_instr = count_short_instr + 1
            if len(step) > MAX_LENGTH:
                MAX_LENGTH = len(step)
                longest_instr = step
    sum_len_ingr = 0
    for ingr in input_ingr_tensors:
        sum_len_ingr = sum_len_ingr + len(ingr)
        if len(ingr) < 30:
            count_short = count_short + 1

    print("Max instruction step length: ", MAX_LENGTH)
    print(idx_to_words(longest_instr, idx2word))
    print("Number of short ingredient lists: ", count_short)
    print("Average ingredient list length:", sum_len_ingr/len(input_ingr_tensors))
    print("Number of long instructions: ", count_short_instr)
    print("Average instruction length:", sum_len_instr/len(instr_tensors))
    print("Training set total size: ", total_steps)


    recipe_pairs_filtered = [(input_ingr_tensors[i], instr_tensors[i]) for i, rec in enumerate(input_ingr_tensors) if i not in rm_indices]
    len(recipe_pairs)-len(recipe_pairs_filtered)

    recipe_step_pairs = []
    for recipe in recipe_pairs_filtered:
        for i, instr_step in enumerate(recipe[1][:-1]):
            recipe_step_pairs.append((instr_step, recipe[1][i+1]))

    print(len(recipe_step_pairs))
    return (recipe_step_pairs, idx2word, word2idx, MAX_LENGTH)

