"""Useful functions for getting transformer language model surprisals."""

import functools
import math
import os
import random
import shutil
import torch

from torch.nn.functional import softmax
from transformers import GPTNeoXForCausalLM, AutoTokenizer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def organize_reading_materials(sentence_path, DATASETS): 

    ## goal is to end up concatenating all the sentences from each dataset
    all_sentences = []
    for dataset in DATASETS:
        file = dataset["name"]
        sheet_name = dataset.get("sheet_name", None)
        # print(f"File: {file}, Sheet Name: {sheet_name}")

        if "geco" in file:
            sentence_colname = "SENTENCE"
            sentenceid_colname = "SENTENCE_ID"
        elif "natstories" in file: 
            sentence_colname = "Sentence"
            sentenceid_colname = "SentNum"
        else:
            raise ValueError(f"Unknown dataset format in file: {file}")

        # Load the file
        file_path = os.path.join(sentence_path, file)
        if file.endswith(".xlsx"):
            sdf = pd.read_excel(file_path, sheet_name=sheet_name)
        elif file.endswith(".csv"):
            sdf = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file}")

        # Extract dataset name and build the DataFrame
        dataset_name = file.split("-")[0]
        temp_df = pd.DataFrame({
            "dataset_name": dataset_name,
            "sentence_number": sdf[sentenceid_colname],
            "sentences": sdf[sentence_colname]
        })
        all_sentences.append(temp_df)


    # Concatenate all datasets into a single DataFrame
    df = pd.concat(all_sentences, ignore_index=True)


    return df

def generate_revisions():
    """Manually generate the list of checkpoints available for Pythia modeling suite"""
    
    # Fixed initial steps
    revisions = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
    
    # Add every 1,000 steps afterward
    revisions.extend(range(2000, 144000, 1000))  # Adjust range as needed
    
    # Format each step as "stepX"
    return [f"step{step}" for step in revisions]


def generate_revisions_test():
    """Manually generate the list of checkpoints available for Pythia modeling suite"""
    
    # Fixed initial steps
    revisions = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 10000,50000, 100000, 143000]
    
    # Format each step as "stepX"
    return [f"step{step}" for step in revisions]

def find_sublist_index(mylist, sublist):
    """Find the first occurence of sublist in list.
    Return the start and end indices of sublist in list"""

    for i in range(len(mylist)):
        if mylist[i] == sublist[0] and mylist[i:i+len(sublist)] == sublist:
            return i, i+len(sublist)
    return None

@functools.lru_cache(maxsize=None)  # This will cache results, handy later...


def run_model(model, tokenizer, sentence, device):
    """Run model, return hidden states and attention"""
    # Tokenize sentence
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    # Run model
    with torch.no_grad():
        output = model(**inputs, output_attentions=True)
        hidden_states = output.hidden_states
        attentions = output.attentions

    return {'hidden_states': hidden_states,
            'attentions': attentions,
            'tokens': inputs}

### ... grab the embeddings for your target tokens
def get_embedding(hidden_states, inputs, tokenizer, target, layer, device):
    """Extract embedding for TARGET from set of hidden states and token ids."""
    
    # Tokenize target
    target_enc = tokenizer.encode(target, return_tensors="pt",
                                  add_special_tokens=False).to(device)
    
    # Get indices of target in input tokens
    target_inds = find_sublist_index(
        inputs["input_ids"][0].tolist(),
        target_enc[0].tolist()
    )

    # Get layer
    selected_layer = hidden_states[layer][0]

    #grab just the embeddings for your target word's token(s)
    token_embeddings = selected_layer[target_inds[0]:target_inds[1]]

    #if a word is represented by >1 tokens, take mean
    #across the multiple tokens' embeddings
    embedding = torch.mean(token_embeddings, dim=0)
    
    return embedding

def count_parameters(model):
    """credit: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model"""
    
    total_params = 0
    for name, parameter in model.named_parameters():
        
        # if the param is not trainable, skip it
        if not parameter.requires_grad:
            continue
        
        # otherwise, count it towards your number of params
        params = parameter.numel()
        total_params += params
    # print(f"Total Trainable Params: {total_params}")
    
    return total_params

def compute_surprisal_old(text,tokenizer,model,device):

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens = True)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Shift logits and labels to align:
    # each row t of logits represents a prediction for label at position t+1
    # so: get rid of last row of logits (which has no corresponding actual label in labels sequence)
    # and: get ride of first item of labels, since there is no corresponding logit to predict it
    # see diagram below: x's are for eliminated elements in each variable
    # input labels: _x_ | __ | __ | __ 
    #                    /    /    /
    # outputlogits:  __ / __ / __ / _x_

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Gather log probs of the actual next tokens
    # (selects elements from log_probs along dim == 2, the vocabulary axis, using labels from
    # shift_labels -- unsqueeze(-1) gives shift_labels a third dimension)
    next_token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Convert to surprisal: -log2(p) by invoking Change of Base Formula 
    # (the next_token_log_probs are in natural log, so we need to change to log_2 by dividing over log_e(2))
    surprisals = -next_token_log_probs / math.log(2)
    
    tokens = tokenizer.convert_ids_to_tokens(shift_labels.squeeze().tolist()) 
    if not len(tokens) == 1:
        token_surprisals = list(zip(tokens, surprisals.squeeze().tolist()))
    else: 
        token_surprisals = []

    return token_surprisals


def compute_token_surprisal(text, tokenizer, model, device):
    """Returns (token, surprisal) for each token in the input."""
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    next_token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    surprisals = -next_token_log_probs / math.log(2)  # base-2 surprisal

    tokens = tokenizer.convert_ids_to_tokens(shift_labels.squeeze().tolist())
    if not len(tokens) == 1:
        token_surprisals = list(zip(tokens, surprisals.squeeze().tolist()))
    else: 
        token_surprisals = []

    return token_surprisals



def clean_up_surprisals(token_surprisals,dataset_name): 
    """dataset_name: string, dictates the way to split tokens"""
    words = []
    current_word = []
    current_surprisals = []

    # Iterate through all tokens and put them together into single words when they 
    # correspond to subword tokens; collect their surprisals for later summing
    
    for i,(token, surprisal) in enumerate(token_surprisals):
        # Treat each dataset differently if necessary
        if dataset_name in ["natstories", "geco"]: 
            if (token.startswith("Ġ")) and (i+1 < len(token_surprisals)):
                words.append(("".join(current_word), current_surprisals))
                current_word = [token]
                current_surprisals = [surprisal]
            elif any(elem in token and current_word for elem in [".","!","?"]) and (token != "..."):
                words.append(("".join(current_word), current_surprisals))
            elif "..." in token: 
                current_word.append(token)
                current_surprisals.append(surprisal)
            elif (token.startswith("Ġ")) and (i+1 == len(token_surprisals)):
                continue
            else:
                current_word.append(token)
                current_surprisals.append(surprisal)
            
    # Remove the word-initial special marker
    final_surprisals = [(i.split("Ġ")[1],j) for i,j in words if i.startswith("Ġ")]


    # Combine the surprisals (by summing) for words with more than one subword token
    # return [(i,np.sum(j)) for i,j in final_surprisals]
    ### TODO: Sean added this to also track number of tokens
    return [(i, np.sum(j), len(j)) for i,j in final_surprisals]

def assign_unique_word_ids_geco(word_surprisals, df_sentence_row):
    row = df_sentence_row
    dataset_name = "geco"
    unique_identifiers = []
    for ix, tup in enumerate(word_surprisals):
        sentence_number = row["sentence_number"]
        word_number = ix + 2 #adjust for zero-indexing (+1) and for skipping first word of sentence (+1)
        unique_identifiers.append(dataset_name + "-" + sentence_number + "-" + str(word_number))
    return unique_identifiers

def batch_compute_surprisal(sentences: list[str], model, tokenizer, device, max_length=512):
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    surprisal_results = []

    for i, sentence in enumerate(sentences):
        ids = input_ids[i]
        lp = log_probs[i]

        # Shift for causal: prediction of token i is at position i-1
        token_surprisals = -lp[range(len(ids) - 1), ids[1:]].cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(ids[1:])

        word_surprisals = list(zip(tokens, token_surprisals))
        surprisal_results.append(word_surprisals)

    return surprisal_results  # List of lists: one entry per sentence

def wrangle_metadata_geco(df_reading, df_materials):
    """Adds the sentence_id metadata to the dataframe containing reading time data."""
    participants = list(set(df_reading["PP_NR"].tolist()))
    gather_dfs = []
    for ppt in participants:
        subp = df_reading[df_reading["PP_NR"]==ppt]
        first_word_idx = 0
        current_trial_num = first_trial_num
        for ix,row in df_materials.iloc[:1600].iterrows():
            sentence = row["SENTENCE"]
            sentence_id = row["SENTENCE_ID"]
            trial_num = subp.iloc[first_word_idx]["TRIAL"]
            last_word_idx = first_word_idx + row["NUMBER_WORDS_SENTENCE"]
            sentence_df = subp.iloc[first_word_idx:last_word_idx] 
            sent_read = " ".join(sentence_df["WORD"].values)
            sent_mats = sentence.replace("  ", " ")
            sent_mats = sent_mats[:-1]
            if sent_read == sent_mats:
                sentence_df["SENTENCE_ID"] = np.repeat(sentence_id,row["NUMBER_WORDS_SENTENCE"])
            elif "\'" in sent_read: #the reading df sometimes adds backslashes to words
                sent_read = sent_read.replace("\'","")
                if len(sent_read) == len(sent_mats): #then probably same sentence
                    sentence_df["SENTENCE_ID"] = np.repeat(sentence_id,row["NUMBER_WORDS_SENTENCE"])
                else:
                    sentence_df["SENTENCE_ID"] = np.nan
            elif '"' in sent_read and not '"' in sent_mats: #the reading df sometimes has extra double quotes
                sent_read = sent_read.replace('"',"")
                if len(sent_read) == len(sent_mats): #then probably same sentence
                    sentence_df["SENTENCE_ID"] = np.repeat(sentence_id,row["NUMBER_WORDS_SENTENCE"])
                else: 
                    sentence_df["SENTENCE_ID"] = np.nan
            elif ("Mr" in sent_read or "Mr" in sent_mats) and (np.abs(len(sent_read) - len(sent_mats)) == 1):
                # probably same sentence, sometimes materials writes out "Mr.s." where reading times df write "Mrs."
                sentence_df["SENTENCE_ID"] = np.repeat(sentence_id,row["NUMBER_WORDS_SENTENCE"]) 
            else: 
                sentence_df["SENTENCE_ID"] = np.nan
            gather_dfs.append(sentence_df)
            # Reset the current index to keep tabs on which word ids to grab from reading times df
            first_word_idx = last_word_idx 
    return pd.concat(gather_dfs)

def clear_model_from_cache(cachepath):
    # local example: "../../../../../../.cache/huggingface/hub/"
    # Head to the specified cache directory 

    folders = os.listdir(cachepath)
    targets = [f for f in folders if f.startswith("models")]
    for mdl in targets:
        shutil.rmtree(os.path.join(cachepath,mdl))

    return



