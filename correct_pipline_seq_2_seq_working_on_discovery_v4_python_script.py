# %%
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import json
import pickle
# import matplotlib.pyplot as plt


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_k, d_model, n_heads, max_len, causal = False):
        
        super().__init__()
        
        # assume d_v = d_k
        self.d_k = d_k
        self.n_heads = n_heads
        
        self.key = nn.Linear(d_model, d_k*n_heads)
        self.query = nn.Linear(d_model, d_k*n_heads)
        self.value = nn.Linear(d_model, d_k*n_heads)
        
        # final linear layer
        self.fc = nn.Linear(d_k* n_heads, d_model)
        
        # causal mask
        # make it so that diagonal is 0 too.. 
        # this way we dont have to shift the inputs to make targets
        
        self.causal = causal
        
        # causal mask if causal has been set to true. 
        if causal:
            
            cm = torch.tril(torch.ones(max_len, max_len))
            
            self.register_buffer(
                "causal_mask",
                cm.view(1,1, max_len, max_len)
            )
            
    def forward(self, q,k,v, pad_mask = None):
        
        q = self.query(q)  # N X T X (hd_k)
        k = self.key(k)  # N X T X (hd_k)
        v = self.value(v)  # N X T X (hd_k)
        
        # in seq-2-seq, it is possible to apply attention where we want to know, 
        # which decoder output should pay attention to which encoder input. 
        # this is a cross-attention part, wehre encoder is connected to the decoder. 
        # k and v comes from the encoder while q (query ) comes from the decoder. 
        # so input to this layer comes from encoder input (k), while outptu shape will be q (since output comes from query in decoder)
        N = q.shape[0]
        T_output = q.shape[1]
        T_input = k.shape[1]
        
        #change the shape to:
        # (N,T,h,d_k) --> (N,h,T,d_k)
        
        # in order for amtrix multiply to work properly
        
        q = q.view(N, T_output, self.n_heads, self.d_k).transpose(1,2)
        k = k.view(N, T_input, self.n_heads, self.d_k).transpose(1,2)
        v = v.view(N, T_input, self.n_heads, self.d_k).transpose(1,2)
        
        # compute attention weights
        # (N,h,T, d_k) x(N,h,d_k, T) --> (N,h,T,T)
        
        attn_scores = q @ k.transpose(-2,-1) / math.sqrt(self.d_k)
        
        if pad_mask is not None:
            
            attn_scores = attn_scores.masked_fill(
                pad_mask[:,None,None,:] == 0, float('-inf'))
            
        # what changes is how we apply thhe causal mask.. 
        if self.causal:
            attn_scores = attn_scores.masked_fill(
                self.causal_mask[:,:,:T_output, :T_input] == 0, float('-inf'))
            
        attn_weights = F.softmax(attn_scores, dim = -1)
        
        
        # compute attention-weighted values
        # (N,h,T,T) x (N,h,T,d_k) --> (N,h, T, d_k)
        A = attn_weights @ v
        
        # reshape it back before final linear layer
        A = A.transpose(1,2)  # (N,T,h,d_k)
        
        A = A.contiguous().view(N,T_output, self.d_k*self.n_heads)   # (N, T, h*d_k)
        
        # projection 
        return self.fc(A)
    
class EncoderBlock(nn.Module):
    
    def __init__(self, d_k, d_model, n_heads, max_len,dropout_prob = 0.1):
        
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal = False)
        self.ann = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(), 
            nn.Linear(d_model*4, d_model),
            nn.Dropout(dropout_prob),
        )
        
        self.dropout = nn.Dropout(p = dropout_prob)
        
        
    def forward(self, x, pad_mask = None):
        
        x = self.ln1(x+self.mha(x,x,x,pad_mask))
        x = self.ln2(x+ self.ann(x))
        x = self.dropout(x)
        return x
    

class DecoderBlock(nn.Module):
    
    def __init__(self, d_k, d_model, n_heads, max_len,dropout_prob = 0.1):
        
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.mha1 = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal = True)  # this layer is cross-attention layer
        self.mha2 = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal = False)
        self.ann = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(), 
            nn.Linear(d_model*4, d_model),
            nn.Dropout(dropout_prob),
        )
        
        self.dropout = nn.Dropout(p = dropout_prob)
        
        
    def forward(self, enc_output, dec_input, enc_mask = None, dec_mask = None):
        
        x = self.ln1(
            dec_input + self.mha1(dec_input, dec_input, dec_input, dec_mask)
        )
        
        # multi-head attention including encoder output
        
        x = self.ln2(x + self.mha2(x, enc_output, enc_output, enc_mask))
        
        x = self.ln3(x+self.ann(x))
        
        x = self.dropout(x)
        
        return x
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len = 2048, dropout_prob = 0.1):
        
        super().__init__()
        
        self.dropout = nn.Dropout(p = dropout_prob)
        
        position = torch.arange(max_len).unsqueeze(1)
        exp_term = torch.arange(0, d_model, 2)
        div_term = torch.exp(exp_term* (-math.log(10000.0)/ d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position* div_term)
        pe[0,:, 1::2] = torch.cos(position* div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        
        # x.shape: N xT X D
        x = x + self.pe[:,:x.size(1), :]
        return self.dropout(x)
    
class Encoder(nn.Module):
    
    def __init__(self, vocab_size, max_len, d_k, d_model, 
                 n_heads, n_layers, dropout_prob):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        
        transformer_blocks = [
            
            EncoderBlock(d_k, 
                         d_model, 
                         n_heads, 
                         max_len,
                         dropout_prob) for _ in range(n_layers)]
        
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.ln = nn.LayerNorm(d_model)
        
    def forward(self, x, pad_mask = None):
        
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for block in self.transformer_blocks:
            
            x = block(x, pad_mask)
            
        x = self.ln(x)
            
        return x
        

class Decoder(nn.Module):
    
    def __init__(self, vocab_size, max_len, d_k, d_model, 
                 n_heads, n_layers, dropout_prob):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        
        transformer_blocks = [
            
            DecoderBlock(d_k, 
                         d_model, 
                         n_heads, 
                         max_len,
                         dropout_prob) for _ in range(n_layers)]
        
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, enc_output, dec_input, enc_mask = None, dec_mask = None):
        
        x = self.embedding(dec_input)
        x = self.pos_encoding(x)
        
        for block in self.transformer_blocks:
            x = block(enc_output, x, enc_mask, dec_mask)
            
        x = self.ln(x)
        x = self.fc(x)  # many - to -many

        return x
            
class Transformer(nn.Module):
    
    def __init__(self, encoder, decoder):
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        
    def forward(self,enc_input, dec_input, enc_mask, dec_mask):
        
        enc_output = self.encoder(enc_input, enc_mask)
        dec_output = self.decoder(enc_output, dec_input, enc_mask, dec_mask)
        
        return dec_output

# %%
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

class DataHandler:
    def __init__(self, model_checkpoint='Helsinki-NLP/opus-mt-en-es', 
                 max_input_length=128, max_target_length=128):
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.tokenizer.add_special_tokens({"cls_token": "<s>"})
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def load_dataset(self, filepath):
        # df = pd.read_csv(filepath, sep="\t", header=None)
        # df = df.iloc[:30000]
        # df.columns = ['en', 'es']
        # df.to_csv('spa.csv', index=None)
        raw_dataset = load_dataset('csv', data_files='spa.csv')
        split = raw_dataset['train'].train_test_split(test_size=0.3, seed=42)
        tokenized_datasets = split.map(
            self.preprocess_function, batched=True,
            remove_columns=split["train"].column_names,
        )
        return tokenized_datasets

    def preprocess_function(self, batch):
        
        # It tokenizes the English sentences (batch['en']) using the tokenizer. 
        # The truncation=True argument ensures that any sentence longer than max_input_length
        # is cut off at that length.
        model_inputs = self.tokenizer(
            batch['en'], max_length=self.max_input_length, truncation=True)
        
        # same thing as above for the labels (tokenizes spanish sentences)
        # use max target length for tokenizing. 
        labels = self.tokenizer(
            batch['es'], max_length=self.max_target_length, truncation=True)
        
        # It stores the tokenized Spanish sentences in model_inputs["labels"]. 
        # This is because for sequence-to-sequence tasks like translation,
        # the labels are the target sentences 
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def prepare_dataloader(self, tokenized_datasets, batch_size=32):
        data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        train_loader = DataLoader(
            tokenized_datasets["train"],
            shuffle=True,
            batch_size=batch_size,
            collate_fn=data_collator
        )
        valid_loader = DataLoader(
            tokenized_datasets["test"],
            batch_size=batch_size,
            collate_fn=data_collator
        )
        return train_loader, valid_loader


# %%
class Trainer:
    def __init__(self, model, criterion, optimizer, device, tokenizer, train_loader, valid_loader, key):
        self.model = model  # this is transformer model 
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        # Add tensorboard writer to Trainer class
        # self.writer = SummaryWriter()
        self.writer = SummaryWriter(log_dir=f'runs/{key}')

    
    
    # def print_number_of_trainable_model_parameters(self):
    #     trainable_model_params = 0
    #     all_model_params = 0
    #     for _, param in self.model.named_parameters():
    #         all_model_params += param.numel()
    #         if param.requires_grad:
    #             trainable_model_params += param.numel()
    #     # print(f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%")
    #     return (f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%")
    
    def train(self, epochs):
        
        def get_total_params(module: torch.nn.Module):
            total_params = 0
            for param in module.parameters():
                total_params += param.numel()
            return total_params

        print('Total parameters in model: {:,}'.format(get_total_params(self.model)))
        
        train_losses = np.zeros(epochs)
        validation_losses = np.zeros(epochs)
        train_perplexity_list = np.zeros(epochs)
        validation_perplexity_list = np.zeros(epochs)
        # print("print_number_of_trainable_model_parameters",
        #       self.print_number_of_trainable_model_parameters()) 
        for it in range(epochs):
            self.model.train()
            t0 = datetime.now()
            train_loss = [] 
            train_loss_scalar, train_correct_scalar, total_train_samples_scalar = 0, 0, 0
            for batch in self.train_loader:
                batch = {k:v.to(self.device) for k,v in batch.items()}
                self.optimizer.zero_grad()
                enc_input = batch['input_ids']
                enc_mask = batch['attention_mask']
                targets = batch['labels']
                dec_input, dec_mask = self.prepare_decoder_inputs(targets)
                outputs = self.model(enc_input, dec_input, enc_mask, dec_mask)
                loss = self.criterion(outputs.transpose(2,1), targets)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
                
                train_loss_scalar +=loss.item()
                _,predicted = torch.max(outputs, dim = 2)
                train_correct_scalar += (predicted == targets).sum().item()
                total_train_samples_scalar += targets.ne(self.tokenizer.pad_token_id).sum().item()
                
                
            train_loss = np.mean(train_loss)
            # test_loss = self.evaluate()
            avg_train_loss_scalar = train_loss_scalar / len(self.train_loader)
            train_accuracy_scalar = train_correct_scalar / total_train_samples_scalar
            train_perplexity_scalar = np.exp(avg_train_loss_scalar)
            
            
            avg_valid_loss_scalar, valid_accuracy_scalar, valid_perplexity_scalar,test_loss = self.evaluate()
            
            dt = datetime.now() - t0
            dt_seconds = dt.total_seconds()  # convert duration to seconds
            
             # Log the metrics to tensorboard
            self.writer.add_scalar('Train Loss', avg_train_loss_scalar, it)
            self.writer.add_scalar('Train Accuracy', train_accuracy_scalar, it)
            self.writer.add_scalar('Train Perplexity', train_perplexity_scalar, it)
            self.writer.add_scalar('Validation Loss', avg_valid_loss_scalar, it)
            self.writer.add_scalar('Validation Accuracy', valid_accuracy_scalar, it)
            self.writer.add_scalar('Validation Perplexity', valid_perplexity_scalar, it)
            self.writer.add_scalar('Epoch Duration', dt_seconds, it)
            
            
            train_losses[it] = train_loss
            validation_losses[it] = test_loss
            
            train_perplexity_list[it], validation_perplexity_list[it] =train_perplexity_scalar, valid_perplexity_scalar
            
            
            
            # Log the duration of this epoch to tensorboard
            
            
            print(f'Epoch {it+1}/{epochs}, Train Loss: {avg_train_loss_scalar:.4f}, \
                  Train Accuracy: {train_accuracy_scalar:.4f}, \
                  Train Perplexity: {train_perplexity_scalar:.4f}, \
                  Validation Loss: {avg_valid_loss_scalar:.4f}, \
                  Validation Accuracy: {valid_accuracy_scalar:.4f}, \
                  Validation Perplexity: {valid_perplexity_scalar:.4f}')
            
            # print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss: .4f}, Test Loss: {test_loss: .4f}, Duration: {dt}')
        self.writer.close()  # Close the writer after logging all losses
        return train_losses, validation_losses, train_perplexity_list, validation_perplexity_list

    def prepare_decoder_inputs(self, targets):
        dec_input = targets.clone().detach()
        dec_input = torch.roll(dec_input, shifts = 1, dims = 1)
        dec_input[:,0] = 65001
        dec_input = dec_input.masked_fill(dec_input==-100, self.tokenizer.pad_token_id )
        dec_mask = torch.ones_like(dec_input)
        dec_mask = dec_mask.masked_fill(dec_input == self.tokenizer.pad_token_id,0)
        return dec_input, dec_mask

    def evaluate(self):
        self.model.eval()
        test_loss = []
        valid_loss, valid_correct, total_valid_samples = 0, 0, 0
        for batch in self.valid_loader:
            batch = {k:v.to(self.device) for k,v in batch.items()}
            enc_input = batch['input_ids']
            enc_mask = batch['attention_mask']
            targets = batch['labels']
            dec_input, dec_mask = self.prepare_decoder_inputs(targets)
            outputs = self.model(enc_input, dec_input, enc_mask, dec_mask)
            loss = self.criterion(outputs.transpose(2,1), targets)
            test_loss.append(loss.item())
            
            
            valid_loss += loss.item()
            _, predicted = torch.max(outputs, dim=2)
            valid_correct += (predicted == targets).sum().item()
            total_valid_samples += targets.ne(self.tokenizer.pad_token_id).sum().item()
        avg_valid_loss = valid_loss / len(self.valid_loader)
        valid_accuracy = valid_correct / total_valid_samples
        valid_perplexity = np.exp(avg_valid_loss)
        # return 
            
        return avg_valid_loss, valid_accuracy, valid_perplexity,np.mean(test_loss)

# %%


def main():
    
    # Create an instance of the data handler
    data_handler = DataHandler()

    # Load the dataset from the file and get the tokenized datasets
    tokenized_datasets = data_handler.load_dataset('spa.csv')

    # Prepare the data loaders
    train_loader, valid_loader = data_handler.prepare_dataloader(tokenized_datasets)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    max_lens = [512]
    model_sizes = [64, 512]
    d_ks =[16]
    # model_sizes =[64]
    n_heads =[4]
    n_layers =[2]
    dropout_prob = [0.5]
    epochs = 20


    results = {}
    
    for max_len in max_lens:
        for d_k in d_ks:
            for model_size in model_sizes:
                for n_head in n_heads:
                    for n_layer in n_layers:
                        for dropout in dropout_prob:
                            
                            key = '_'.join(str(x) for x in (max_len, d_k, model_size,n_head,n_layer,dropout))
                            print("Running experiment with configuration:", key)

            
            
                            encoder = Encoder(vocab_size=data_handler.tokenizer.vocab_size+1,
                                            max_len = max_len,
                                            d_k = d_k,
                                            d_model = model_size,
                                            n_heads = n_head,
                                            n_layers = n_layer,
                                            dropout_prob = dropout)
                            encoder.to(device)

                            decoder = Decoder(vocab_size=data_handler.tokenizer.vocab_size+1,
                                            max_len =max_len,
                                            d_k=d_k,
                                            d_model=model_size,
                                            n_heads=n_head,
                                            n_layers=n_layer,
                                            dropout_prob=dropout)
                            decoder.to(device)

                            transformer = Transformer(encoder, decoder)

                            criterion = torch.nn.CrossEntropyLoss(ignore_index = -100)
                            optimizer = torch.optim.Adam(transformer.parameters())

                            trainer = Trainer(transformer, criterion, optimizer, device, data_handler.tokenizer, train_loader, valid_loader, key)
                            train_losses, validation_losses, train_perplexity_list, validation_perplexity_list = trainer.train(epochs=epochs)
                            
                            results[key] = {
                                "train_losses_list":train_losses,
                                "validation_losses_list":validation_losses,
                                "model_size": model_size,
                                "d_k": d_k,
                                "max_len": max_len,
                                "n_head": n_head,
                                "n_layer": n_layer,
                                "dropout": dropout,
                            }
                            
                            
main()

# %%


# %%


# %%



