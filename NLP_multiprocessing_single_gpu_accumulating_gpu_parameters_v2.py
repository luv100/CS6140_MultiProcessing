import math
import torch
import torch.nn as nn
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import time
import math
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import json
import pickle

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(Encoder, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # Need (S, N) format for encoder.
        src = src.t()
        src = self.encoder(src) * math.sqrt(self.ninp)
        return self.pos_encoder(src)

class Decoder(nn.Module):
    def __init__(self, ntoken, ninp):
        super(Decoder, self).__init__()
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp):
        # Need batch dimension first for output of pipeline.
        return self.decoder(inp).permute(1, 0, 2)    
    
def run_worker(writer,model_size, batch_size, bptt, optimizer_type, lr_schedule, precision,nlayers,nheads, dropout,n_epochs=10):
    
    def print_with_rank(msg):
        print('[Single GPU]: {}'.format(msg))

    # Initialize tokenizer, vocab, and data
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"]) 

    def data_process(raw_text_iter):
      data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
      return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def batchify(data, bsz):
        # Divide the dataset into `bsz` parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the `bsz` batches.
        return data.view(bsz, -1).t().contiguous().to(device)

    # batch_size = 20
    eval_batch_size = 10
    train_data = batchify(train_data, batch_size)
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size)
    
    # bptt = 35
    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data.t(), target

    ntokens = len(vocab) # the size of vocabulary
    emsize = model_size # embedding dimension, was 4096
    nhid = model_size # the dimension of the feedforward network model in nn.TransformerEncoder, was 4096
    nlayers = nlayers # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder, was 8
    nhead = nheads # the number of heads in the Multihead Attention models, was 16
    dropout = dropout # the dropout value remains the same

    
    model = nn.Sequential(
        Encoder(ntokens, emsize, dropout).to(device),
        nn.TransformerEncoder(nn.TransformerEncoderLayer(emsize, nhead, nhid, dropout), nlayers).to(device),
        Decoder(ntokens, emsize).to(device)
    )
    
    # switch to half precision if requested
    if precision == 'half':
        model = model.half()

    def get_total_params(module: torch.nn.Module):
        total_params = 0
        for param in module.parameters():
            total_params += param.numel()
        return total_params

    print_with_rank('Total parameters in model: {:,}'.format(get_total_params(model)))

    criterion = nn.CrossEntropyLoss()
    # lr = 5.0 # learning rate
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
     # Create lr_scheduler based on lr_schedule
    if lr_schedule == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    elif lr_schedule == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Convert max memory allocated to MBs
    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    # Calculate runtime
    start_time = time.time()
    # The rest of your training code here...
    def train():
        model.train() 
        total_loss = 0.
        start_time = time.time()
        ntokens = len(vocab)
        gpu_memory_usage = []
        num_batches = 0

        # Train only for 50 batches to keep script execution time low.
        nbatches = min(50 * bptt, train_data.size(0) - 1)

        for batch, i in enumerate(range(0, nbatches, bptt)):
            data, targets = get_batch(train_data, i)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.reshape(-1, ntokens), targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            log_interval = 10
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print_with_rank('| epoch {:3d} | {:5d}/{:5d} batches | '
                    'lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, nbatches // bptt, scheduler.get_last_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

            # Store current GPU memory usage
            gpu_memory_usage.append(torch.cuda.memory_allocated(device) / 1024 ** 2)  # Convert bytes to megabytes

        #The Average Loss per Batch: This is the mean of the losses for each batch in an epoch. 
        # Each batch consists of a certain number of examples (determined by your batch size).

        # In most cases, reporting the average loss per batch (option 2) is more meaningful. 
        return total_loss / num_batches, gpu_memory_usage

    # ... training process ...
    
    # Convert gpu_memory_usage to MBs
    # gpu_memory_usage_MBs = [usage / (1024 ** 2) for usage in gpu_memory_usage]
    
    def evaluate(eval_model, data_source):
        eval_model.eval() # Turn on the evaluation mode
        total_loss = 0.
        ntokens = len(vocab)
        # Evaluate only for 50 batches to keep script execution time low.
        nbatches = min(50 * bptt, data_source.size(0) - 1)
        with torch.no_grad():
            for i in range(0, nbatches, bptt):
                data, targets = get_batch(data_source, i)
                output = eval_model(data)
                output_flat = output.reshape(-1, ntokens)
                # output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets.to(device)).item()
        return total_loss / (len(data_source) - 1)

        # Initialize the tensorboard writer
    

    best_val_loss = float("inf")
    epochs = n_epochs # The number of epochs
    best_model = None

    gpu_usage_per_epoch = []
    run_time_per_epoch = []
    validation_perplexity_per_epoch = []

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loss, gpu_usage = train()

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # train_loss = calculate_train_loss()  # replace this with your method to calculate train loss

        val_loss = evaluate(model, val_data)
        epoch_end_time = time.time()

        # Compute the perplexities
        train_perplexity = math.exp(train_loss)
        valid_perplexity = math.exp(val_loss)

        # Log the losses, perplexities, learning rate and time
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', val_loss, epoch)
        writer.add_scalar('Perplexity/train', train_perplexity, epoch)
        writer.add_scalar('Perplexity/valid', valid_perplexity, epoch)
        writer.add_scalar('Learning Rate', lr, epoch)
        writer.add_scalar('Time/epoch', epoch_end_time - epoch_start_time, epoch)
        writer.flush()
        print_with_rank('-' * 89)
        print_with_rank('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        print_with_rank('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()
        gpu_usage_per_epoch.append(gpu_usage)  # Append GPU usage data for this epoch
        run_time_per_epoch.append(epoch_end_time-epoch_start_time)
        validation_perplexity_per_epoch.append((val_loss, math.exp(val_loss)))

    end_time = time.time()
    runtime = end_time - start_time  # in seconds

    test_loss = evaluate(best_model, test_data)
    print_with_rank('=' * 89)
    print_with_rank('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print_with_rank('=' * 89)

    

    return torch.cuda.max_memory_allocated(), runtime, gpu_usage_per_epoch,run_time_per_epoch,validation_perplexity_per_epoch  # Return peak GPU memory usage, total runtime, and GPU memory usage for each epoch    
# Main execution
if __name__=="__main__":
    
    # # Define lists of parameters
    # model_sizes = [4, 16]
    # batch_sizes = [16, 32]
    # bptts = [16, 32]
    # optimizer_types = ['SGD', 'Adam']
    # lr_schedules = ['step', 'cosine']
    # precisions = ['full', 'half']
    
        # Define lists of parameters
    
    # model_sizes = [128, 256, 512, 1024, 2048,4096, 8192]
    # batch_sizes = [4,8,16, 32]
    # bptts = [16,32]
    # optimizer_types = ['SGD', 'Adam']
    # lr_schedules = ['step', 'cosine']
    # precisions = ['full', 'half']
    # model_sizes = [512, 1024]
    # dropouts = [0.5,0.4,0.3]

    model_sizes = [512]
    dropouts = [0.5]


    batch_sizes = [20]
    nlayers = 8
    nheads = 16
    bptts = [16]
    optimizer_types = ['Adam']
    lr_schedules = ['cosine']
    precisions = ['full']
    writer = SummaryWriter()
    
    # Run experiments and store results
    results = {}
    for model_size in model_sizes:
        for batch_size in batch_sizes:
            for bptt in bptts:
                for optimizer_type in optimizer_types:
                    for lr_schedule in lr_schedules:
                        for precision in precisions:
                            for dropout in dropouts:
                                # key = (model_size, batch_size, bptt, optimizer_type, lr_schedule, precision)
                                key = '_'.join(str(x) for x in (model_size, batch_size, bptt, optimizer_type, lr_schedule, precision))
                                print("results[key]",results)
                                
                                results[key] = {}
                                max_memory_allocated, runtime, gpu_memory_usage, run_time_per_epoch,validation_perplexity_per_epoch \
                                    = run_worker(writer,model_size, batch_size, bptt, optimizer_type, lr_schedule, precision,nlayers,nheads,dropout, n_epochs=1000)
                                    # model_size, batch_size, bptt, optimizer_type, lr_schedule, precision,nlayers,nheads, n_epochs=10
                                # results[key]['max_memory_allocated'] = max_memory_allocated
                                # results[key]['runtime'] = runtime
                                # results[key]['gpu_memory_usage'] = gpu_memory_usage
                                # results[key]['run_time_per_epoch'] = run_time_per_epoch
                                # results[key]['validation_perplexity_per_epoch'] = validation_perplexity_per_epoch
                                
                                results[key] = {
                                "model_size": model_size,
                                "batch_size": batch_size,
                                "bptt": bptt,
                                "optimizer_type": optimizer_type,
                                "lr_schedule": lr_schedule,
                                "precision": precision,
                                "dropout": dropout,
                                "peak_memory": max_memory_allocated,
                                "runtime": runtime,
                                "gpu_usage": gpu_memory_usage,
                                "run_time_per_epoch": run_time_per_epoch,
                                "validation_perplexity_per_epoch": validation_perplexity_per_epoch
                            }
                            
    # print("results", results)
    # print("results", results)
    writer.close()  # Close the writer after logging all necessary data
    # Save results to a json file
    with open('results.json', 'w') as f:
        json.dump(results, f)

    # Save results to a pickle file
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)