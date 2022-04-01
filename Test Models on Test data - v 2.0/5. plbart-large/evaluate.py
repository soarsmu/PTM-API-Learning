import torch
import os, sys
import pandas as pd
from bleu_score import sentence_bleu
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

#Tokenizer
from transformers import RobertaTokenizerFast

#Encoder-Decoder Model
from transformers import EncoderDecoderModel

from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from dataclasses import dataclass, field
from typing import Optional
from datasets import Dataset
import random

# Set the path to the data folder, datafile and output folder and files
root_folder = os.getcwd()
data_folder = os.path.abspath(os.path.join(root_folder, 'datasets'))
model_folder = os.path.abspath(os.path.join(root_folder, 'model/2022013001'))

# Datafiles names containing training and test data
test_filename='test.csv'
data = 'evaluation.csv'
outputfile = 'submission.csv'
outputfile_path = os.path.abspath(os.path.join(root_folder,outputfile))
testfile_path = os.path.abspath(os.path.join(data_folder,test_filename))
datapath = os.path.abspath(os.path.join(data_folder,data))

TRAIN_BATCH_SIZE = 4
MAX_LEN = 512
batch_size = TRAIN_BATCH_SIZE

# Generate a text using beams search
def generate_results(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["annotation"], padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = model.generate(input_ids, attention_mask=attention_mask,
                                  num_beams=15,
                                  repetition_penalty=3.0, 
                                  length_penalty=2.0, 
                                  num_return_sequences = 1
    )

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

def get_response(input_text,num_return_sequences,num_beams):
    batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

def get_bs(result):
    reference = pd.read_csv(datapath, header=0)['api_seq'].tolist()
    scores = []
    for i in range(len(result)):
        scores.append(sentence_bleu([reference[i].split()], result[i].split()))
        # print(i, scores[i])
    
    avg_bs = (sum(scores)/len(scores))*100
    print('Average Bleu Score: ', avg_bs)
    return avg_bs

if __name__ == "__main__":
    print("===================================MAIN FUNCTION STARTED========================")
    
    if os.path.exists(datapath):
        df = pd.read_csv(datapath, header=0)
        temp_df = pd.read_csv(testfile_path, header=0)
        num_beams = 30
        num_return_sequences = 30
        result = []
        if df.shape[0] == temp_df.shape[0]:
            old_stdout = sys.stdout # backup current stdout
            sys.stdout = open(os.devnull, "w")
            model_name = os.path.abspath(os.path.join(root_folder,"model/aux_model"))
            torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            tokenizer = PegasusTokenizer.from_pretrained(model_name)
            model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
            

            pred_str_bs = df['api_seq'].tolist()
            count = 0
            
            sys.stdout = old_stdout

            for pred in pred_str_bs:
                result.append(get_response(pred,num_return_sequences,num_beams)[21])
                count += 1
                print(count, end="/10000; ")
                # if count >= 2:
                #     break
            pd.DataFrame({'col':result}).to_csv(outputfile_path, encoding='utf-8')
        bleu_score = get_bs(result)

    if not os.path.exists(datapath):
        df = pd.read_csv(datapath, header=0)
        temp_df = pd.read_csv(testfile_path, header=0)
        if df.shape[0] != temp_df.shape[0]:
            df=pd.read_csv(testfile_path, header=0)
            test_data=Dataset.from_pandas(df.head(200))
            checkpoint_path = os.path.abspath(os.path.join(model_folder,'final-checkpoint'))

            old_stdout = sys.stdout # backup current stdout
            sys.stdout = open(os.devnull, "w")
            tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint_path)
            model = EncoderDecoderModel.from_pretrained(checkpoint_path)
            model.to("cuda")
            sys.stdout = old_stdout

            # Generate predictions using beam search
            results = test_data.map(generate_results, batched=True, batch_size=batch_size, remove_columns=["annotation"])
            pred_str_bs = results["pred"]
            bleu_score = get_bs(pred_str_bs)