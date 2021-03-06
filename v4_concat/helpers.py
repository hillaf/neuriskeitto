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


def get_ingr_tensors(word2idx):
    # Load preprocessed data
    with open('project_data/project_train_data_ingr.json') as json_file:
        all_recs_ingr = json.load(json_file)
    # To idx representation
    input_ingr = []
    for recipe in all_recs_ingr:
        # Leave out <LN> helper tokens
        input_ingr.append([word2idx[w] for w in recipe if w != '<LN>'])
    # Create tensors
    input_ingr_tensors = [torch.tensor(rec).view(-1, 1) for rec in input_ingr]
    return input_ingr_tensors


def get_instr_tensors(word2idx):
    # Load data
    with open('project_data/project_train_data_instr.json') as json_file:
        all_recs_instr = json.load(json_file)
    # Preprocess: to idx representation & create tensors
    instr_tensors = []
    for recipe in all_recs_instr:
        step_tensors = []
        for step in recipe:
            idx_list = [word2idx[w] for w in step]
            step_tensors.append(torch.tensor(idx_list).view(-1, 1))
        instr_tensors.append(step_tensors)
    return instr_tensors


def filter_ingredients(recipes):
    sum_len_ingr = 0
    count_short = 0
    for rec in recipes:
        ingr = rec[0]
        sum_len_ingr = sum_len_ingr + len(ingr)
        if len(ingr) < 30:
            count_short = count_short + 1
    print("Number of short ingredient lists: ", count_short)
    print("Average ingredient list length:", sum_len_ingr/len(recipes))
    print("No ingredients filtered")


def filter_instructions(recipes, limit):
    sum_len_instr = 0
    count_short_instr = 0
    MAX_LENGTH = 0
    total_steps = 0
    longest_instr = []
    rm_indices = set()
    for i, recipe in enumerate(recipes):
        rec = recipe[1]
        # filter out recipes that have only one step
        if len(rec) < 2:
            rm_indices.add(i)
        for step in rec:
            total_steps = total_steps + 1
            sum_len_instr = sum_len_instr + len(step)
            if len(step) > limit:
                # filter out steps that are too long
                rm_indices.add(i)
                count_short_instr = count_short_instr + 1
            elif len(step) > MAX_LENGTH:
                # longest step that is not filtered
                MAX_LENGTH = len(step)
                longest_instr = step
    print("Max instruction step length: ", MAX_LENGTH)
    print("Number of long instructions: ", count_short_instr)
    print("Average instruction length:", sum_len_instr/len(recipes))
    print("Total instruction steps: ", total_steps)
    recipe_pairs_filtered = [recipes[i] for i, rec in enumerate(recipes) if i not in rm_indices]
    print("Recipes filtered: ", len(recipes)-len(recipe_pairs_filtered))
    print("Recipes left after filtering: ", len(recipe_pairs_filtered))
    return (recipe_pairs_filtered, MAX_LENGTH)


def get_instruction_steps(recipes):
    recipe_step_pairs = []
    for recipe in recipes:
        for i, instr_step in enumerate(recipe[1][:-1]):
            recipe_step_pairs.append((instr_step, recipe[1][i+1]))
    print("Recipe step pairs: ", len(recipe_step_pairs))
    return recipe_step_pairs



def get_instruction_steps_with_ingredients(recipes):
    MAX_LENGTH = 0
    recipe_step_pairs = []
    for recipe in recipes:
        for i, instr_step in enumerate(recipe[1][:-1]):
            ingr_idx_list =  [j[0] for j in recipe[0].tolist()]
            idx_list = ingr_idx_list[:-1]
            first_instr_step = [j[0] for j in instr_step.tolist()]
            idx_list.extend(first_instr_step[1:])
            idx_list = torch.tensor(idx_list).view(-1, 1)
            recipe_step_pairs.append((idx_list, recipe[1][i+1]))
            if len(idx_list) > MAX_LENGTH:
                MAX_LENGTH = len(idx_list)
    print("Recipe step pairs: ", len(recipe_step_pairs))
    print("New max length: ", MAX_LENGTH)
    return recipe_step_pairs, MAX_LENGTH


def get_first_steps(recipes):
    recipe_first_steps = []
    for rec in recipes:
        recipe_first_steps.append(rec[0])
    return recipe_first_steps


def get_instr_tensors_with_ingredients(word2idx, ingredients):
    # Load data
    with open('project_data/project_train_data_instr.json') as json_file:
        all_recs_instr = json.load(json_file)
    # Preprocess: to idx representation & create tensors
    instr_tensors = []
    for i, recipe in enumerate(all_recs_instr):
        step_tensors = []
        for step in recipe:
            ingr_idx_list = [j[0] for j in ingredients[i].tolist()]
            instr_idx_list = [word2idx[w] for w in step]
            idx_list = ingr_idx_list[:-1]
            instr_idx_list = instr_idx_list[1:]
            idx_list.extend(instr_idx_list)
            step_tensors.append(torch.tensor(idx_list).view(-1, 1))
        instr_tensors.append(step_tensors)
    return instr_tensors


def get_tensor_data(limit=70):
    # Load index mappings created in preprocessing
    with open('project_data/project_train_data_word2idx.json') as json_file:
        word2idx = json.load(json_file)
    with open('project_data/project_train_data_idx2word.json') as json_file:
        idx2word = json.load(json_file)

    # Get tensor representation
    input_ingr_tensors = get_ingr_tensors(word2idx)
    input_instr_tensors = get_instr_tensors(word2idx)
    recipes = [(input_ingr_tensors[i], input_instr_tensors[i]) for i, rec in enumerate(input_ingr_tensors)]

    # Filter data
    filter_ingredients(recipes)
    recipes_filtered, MAX_LENGTH = filter_instructions(recipes, limit)

    # Step pairs
    # recipe_step_pairs = get_instruction_steps(recipes_filtered)
    recipe_step_pairs, MAX_LENGTH = get_instruction_steps_with_ingredients(recipes_filtered)
    print(recipe_step_pairs[0][1])
    print(idx_to_words(recipe_step_pairs[0][0], idx2word))
    print(idx_to_words(recipe_step_pairs[0][1], idx2word))


    return (recipe_step_pairs, idx2word, word2idx, MAX_LENGTH)

