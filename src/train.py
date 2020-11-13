from submodels import TransformerModel, PositionalEncoding
import os
from utils import data_paths
from torch import nn, optim
import torch


def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), 1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def run_training(d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                dim_feedforward: int = 2048, dropout: float = 0.1, vocab_size=32000, tokenizer="tr",
                dataset="antoloji", style=0., rhyme=0., rl=False, batch_size=32, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                             num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                             dropout=dropout, vocab_size=vocab_size, tokenizer=tokenizer,
                             batch_size=batch_size).to(device)
    model_path = os.path.join("training", dataset, model.config_string, "transformer")
    iter = 0
    if os.path.exists(model_path):
        models = sorted(os.listdir(model_path), key=lambda x: os.stat(os.path.join(model_path, x)).st_mtime)
        if len(models) > 0:
            model.load(os.path.join(model_path, models[-1]))
            iter = int(models[-1].partition(".pt")[0].partition("iter")[2])
        else:
            model.save(os.path.join(model_path, "iter{}.pt".format(iter)))
    else:
        os.makedirs(model_path)
        model.save(os.path.join(model_path, "iter{}.pt".format(iter)))
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=3)
    lr = 5.0  # learning rate
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    for epoch in range(epochs):
        print("Beginning epoch {}.".format(epoch + 1))
        total_loss = 0.
        src, tgt = model.batch_generator(data_paths[dataset]["src"]), model.batch_generator(data_paths[dataset]["tgt"])
        for src_batch, tgt_batch in zip(src, tgt):
            model.train()
            optimizer.zero_grad()
            src_input, src_mask = src_batch
            tgt_input, tgt_mask = tgt_batch
            tgt_output = tgt_input[1:, :]
            decoder_input = tgt_input[:-1, :]
            output = model.forward(src_input, decoder_input)
            output_flat = output.view(-1, vocab_size)
            tgt_flat = tgt_output.flatten()
            loss = criterion(output_flat, tgt_flat)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            decoded = model.predict_inference_src("Nazım Hikmet vatan hainliğine devam ediyor hala.", max_len=128)
            # TODO shift target by one, without start token for training.
            print(decoded)
            total_loss += loss.item()
            print(total_loss)
            iter += 1
            print(iter)
            if iter % 5000 == 0:
                print("Saving model after {} iterations". format(iter))
                model.save(os.path.join(model_path, "iter{}.pt".format(iter)))

        scheduler.step()



if __name__ == "__main__":
    run_training(d_model=128, dim_feedforward=128, num_decoder_layers=2, num_encoder_layers=2, batch_size=4)
