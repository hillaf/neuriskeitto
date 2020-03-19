#!/usr/bin/env python
# coding: utf-8

# Data loading and preprocessing


import json
from collections import Counter
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk


# Load data from files
allrecipes = [json.loads(line) for line in open('original_data/allrecipes-recipes.json', 'r')]
bbc = [json.loads(line) for line in open('original_data/bbccouk-recipes.json', 'r')]
epi = [json.loads(line) for line in open('original_data/epicurious-recipes.json', 'r')]
cookstr = [json.loads(line) for line in open('original_data/cookstr-recipes.json', 'r')]


# Unify keysets

#print(len(allrecipes))
#print(len(bbc))
#print(len(epi))
#print(len(cookstr))

#print(allrecipes[0].keys())
#print(bbc[0].keys())
#print(epi[0].keys())
#print(cookstr[0].keys())

allrecipes_data = [{'name': rec['title'],
                    'ingredients': rec['ingredients'],
                    'instructions': rec['instructions']} for rec in allrecipes]

# detailed instructions?
bbc_data = [{'name': rec['title'],
            'ingredients': rec['ingredients'],
            'instructions': rec['instructions']} for rec in bbc]


epi_data = [{'name': rec['hed'],
            'ingredients': rec['ingredients'],
            'instructions': rec['prepSteps']} for rec in epi if 'ingredients' in rec]

# detailed ingredients?
cookstr_data = [{'name': rec['title'],
                'ingredients': rec['ingredients'],
                'instructions': rec['instructions']} for rec in cookstr]





# Combine training data and filter out bad items
# Keep cookstr as a test set
all_data = allrecipes_data + bbc_data + epi_data
filter_empty = set()

# Sanity check
names = [item['name'] for item in all_data]
counter = Counter(names)
counter.most_common(10)

for i, r in enumerate(all_data):
    if r['name'] == '' or r['name'] =="Johnsonville® Three Cheese Italian Style Chicken Sausage Skillet Pizza":
        filter_empty.add(i)
    if len(r['ingredients'])==0:
        filter_empty.add(i)
    if len(r['instructions'])==0:
        filter_empty.add(i)

print("Recipes filtered", len(filter_empty))

dev_data = [item for i, item in enumerate(all_data) if i not in filter_empty]

print("Total number of recipes in the dataset:", len(dev_data))



# Extract quantity and quantity variable information

ingredient_data = [rec['ingredients'] for rec in dev_data]

amount_r = r'((\d{1,2}|½|¼)(\/\d)?(\s\d\/\d)?(\s\(\d{1,2}\sounce\))?)'
measure_r = r'(cup(s)?|teaspoon(s)?|packet(s)?|box(es)?|package(s)?|tablespoon(s)?|ounce(s)?|pinch|square(s)?|pound(s)?|slice(s)?|bunch|cube(s)?|can(s)?|pint(s)?|drop(s)?|quart(s)?)'
random_notes_r = r'(\(.*\))'

parsed_ingredients_per_recipe = []
for rec in ingredient_data:
    parsed_ingredients = {}
    for ing in rec:
        amount = re.search(amount_r, ing)
        measure = re.search(measure_r, ing)
        content = re.sub(amount_r, "", ing, count=1)
        content = re.sub(measure_r, "", content, count=1)
        content = re.sub(random_notes_r, "", content)
        content = content.strip()
        if amount and measure:
            amount_re = re.sub(random_notes_r, "", amount.group(0))
            parsed_ingredients[content] = (amount_re, measure.group(0))
        elif amount:
            amount_re = re.sub(random_notes_r, "", amount.group(0))
            parsed_ingredients[content] = (amount_re, "")
        elif measure:
            parsed_ingredients[content] = ("", measure.group(0))
        else:
            parsed_ingredients[content] = ("", "")
    parsed_ingredients_per_recipe.append(parsed_ingredients)

print(len(parsed_ingredients_per_recipe))
print(parsed_ingredients_per_recipe[0])




# Get ingredient names
ingr_names_per_recipe = []
for ingr in parsed_ingredients_per_recipe:
    ingr_names = [key.strip() for key in ingr.keys()]
    ingr_names_per_recipe.append(ingr_names)

ingr_names_comma = r'.*,'
ingr_names_end = r'\s(\S+)$'

simple_ingr_names = []
for rec in ingr_names_per_recipe:
    simple_rec = []
    for i in rec:
        name = re.search(ingr_names_comma, i)
        if not name:
            name = re.search(ingr_names_end, i)
        if name:
            simple_rec.append(name.group(0).replace(',','').strip())
    simple_ingr_names.append(simple_rec)
print(simple_ingr_names[0])


# Create a list and set of all the ingredients together
list_of_ingredients = []
for rec in simple_ingr_names:
    list_of_ingredients.extend(rec)

ingr_counts = Counter(list_of_ingredients)
print(ingr_counts.most_common(30))

set_of_ingredients = set(list_of_ingredients)
print("Number of simplified ingredients: ", len(set_of_ingredients))


# Preprocess instructions

import spacy # import the spaCy module
nlp = spacy.load("en") # load the English model
from spacy import displacy
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

instruction_data = [rec['instructions'] for rec in dev_data]

random_notes_r = r'(\(.*?\))'

parsed_instructions_per_recipe = []
for rec in instruction_data[0:10]:
    parsed_instructions = []
    for instr in rec:
        content = re.sub(random_notes_r, "", instr)
        parsed_instructions.append(content)
        #doc = nlp(content)
        #for sent in doc.sents:
        #    tokenized_sent = []
        #    for token in sent: # iterate over every token
        #        if token.pos_ == 'VERB' or token.dep_ == 'punct' or token.dep_ == 'cc' or token.dep_ == 'ROOT' or token.dep_ == 'adj'or token.dep_ == 'conj' or token.dep_ == 'nsubj' or token.dep_ == 'nobj' or token.dep_ == 'pobj' or token.dep_ == 'dobj' or token.dep_ == 'prep':
        #            tokenized_sent.append(token)
        #        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
        #    parsed_instructions.append(tokenized_sent)
    parsed_instructions_per_recipe.append(parsed_instructions)

print(instruction_data[3])
print(parsed_instructions_per_recipe[3])




# Preprocessing for instructions - slow!
word2idx = {}
idx2word = {}

def preprocess_recipes(ingr_data, instr_data):
    tokenized_instr = []
    tokenized_ingr = []
    n_words = 0
    for i, rec in enumerate(ingr_data):
        if i % 5000 == 0:
            print("Preprocessing recipe ", i)
        recipe_ingr = []
        for line in ingr_data[i]:
            sents = sent_tokenize(line)
            words = []
            for s in sents:
                words_tokenized = word_tokenize(s)
                words.append(words_tokenized)
                for w in words_tokenized:
                    if w not in word2idx:
                        word2idx[w] = n_words
                        idx2word[n_words] = w
                        n_words = n_words + 1
            recipe_ingr.append(words)
        tokenized_ingr.append(recipe_ingr)
        recipe_instr = []
        for line in instr_data[i]:
            sents = sent_tokenize(line)
            words = []
            for s in sents:
                words_tokenized = word_tokenize(s)
                words.append(words_tokenized)
                for w in words_tokenized:
                    if w not in word2idx:
                        word2idx[w] = n_words
                        idx2word[n_words] = w
                        n_words = n_words + 1
            recipe_instr.append(words)
        tokenized_instr.append(recipe_instr)
    return (tokenized_ingr, tokenized_instr)



preprocess = True
instruction_data = [rec['instructions'] for rec in dev_data]


# Either preprocess the data or load previously a saved file
if preprocess:
    train_data_ingr, train_data_instr = preprocess_recipes(simple_ingr_names, instruction_data)
    # Save preprocessed data to a file
    #with open('project_data/project_train_data_tokenized_ingr.json', 'w') as outfile:
    #    json.dump(train_data_ingr, outfile)
    #with open('project_data/project_train_data_tokenized_instr.json', 'w') as outfile:
    #    json.dump(train_data_instr, outfile)
    #with open('project_data/project_train_data_word2idx.json', 'w') as outfile:
    #    json.dump(word2idx, outfile)
    #with open('project_data/project_train_data_idx2word.json', 'w') as outfile:
    #    json.dump(idx2word, outfile)
else:
    with open('project_data/project_train_data_tokenized_ingr.json') as json_file:
        train_data_ingr = json.load(json_file)
    with open('project_data/project_train_data_tokenized_instr.json') as json_file:
        train_data_instr = json.load(json_file)
    with open('project_data/project_train_data_word2idx.json') as json_file:
        word2idx = json.load(json_file)
    with open('project_data/project_train_data_idx2word.json') as json_file:
        idx2word = json.load(json_file)
print(train_data_ingr[0])
print(train_data_instr[0])
print(word2idx['tomato'])
print(idx2word[0])


# Preprocess: add helper tags

all_recs_ingr = []
for recipe in train_data_ingr:
    rec_words = ['<SOS>']
    for sents in recipe:
        rec_words.extend([w for sent in sents for w in sent])
        rec_words.append('<LN>')
    rec_words.append('<EOS>')
    all_recs_ingr.append(rec_words)
print(all_recs_ingr[0])

all_recs_instr = []
for recipe in train_data_instr:
    rec_steps = []
    for step in recipe:
        rec_words = ['<SOS>']
        rec_words.extend([w for sent in step for w in sent])
        rec_words.append('<EOS>')
        rec_steps.append(rec_words)
    all_recs_instr.append(rec_steps)
print(all_recs_instr[0])

print(len(set_of_ingredients))
print(len(word2idx))
print(len(idx2word))


# Add helper tokens
n_words = len(word2idx)
helper_tokens = ['<SOS>', '<LN>', '<EOS>']

for hw in helper_tokens:
    word2idx[hw] = n_words
    idx2word[n_words] = hw
    n_words = n_words + 1


# Save preprocessed data to file
with open('project_data/project_train_data_ingr.json', 'w') as outfile:
    json.dump(all_recs_ingr, outfile)
with open('project_data/project_train_data_instr.json', 'w') as outfile:
    json.dump(all_recs_instr, outfile)
with open('project_data/project_train_data_word2idx.json', 'w') as outfile:
    json.dump(word2idx, outfile)
with open('project_data/project_train_data_idx2word.json', 'w') as outfile:
    json.dump(idx2word, outfile)





