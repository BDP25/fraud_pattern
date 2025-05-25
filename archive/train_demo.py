import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import json

# tokenizer & dataset
class TreatmentSequenceDataset(Dataset):
    def __init__(self, sequences, token2idx, max_len=30):
        self.token2idx = token2idx
        self.max_len = max_len
        self.encoded = [self.encode(seq) for seq in sequences]

    def encode(self, seq):
        tokens = [self.token2idx.get(t, 0) for t in seq]
        if len(tokens) < self.max_len:
            tokens += [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        return tokens

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded[idx], dtype=torch.long)
        return x, x 


def build_tokenizer(sequences):
    vocab = set()
    for seq in sequences:
        cleaned = [t for t in seq if isinstance(t, str) and t.strip()]
        vocab.update(cleaned)
    token2idx = {token: idx + 1 for idx, token in enumerate(sorted(vocab))}
    token2idx['<PAD>'] = 0
    return token2idx


# model
class TransformerAutoencoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4, max_len=30):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, d_model))
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x) + self.pos_encoder[:, :x.size(1)]
        out = self.transformer(emb, emb)
        logits = self.decoder(out)
        return logits



# train
def train_model(model, dataloader, epochs=2, lr=1e-3):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

    return model


# eval
def evaluate(model, dataloader):
    device = next(model.parameters()).device
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    seq_losses = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss.view(x.size(0), x.size(1))
            seq_loss = loss.mean(dim=1).cpu().numpy()
            seq_losses.extend(seq_loss)

    return np.array(seq_losses)



def main():
    # check features
    feature_file = "./data/doctor_features.pkl"
    assert os.path.exists(feature_file), "Missing features, invoke feature_extractor.py first!"

    # load sequences
    with open(feature_file, 'rb') as f:
        data = pickle.load(f)
        doctor_features = data['doctor_features']

    # group features by specialty
    specialty_sequences = defaultdict(list)
    doctor_sequences = defaultdict(list)
    for doc_id, feat in doctor_features.items():
        spec = feat.get('specialty', 'unknown')
        for seq in feat.get('all_sequences', []):
            if len(seq) > 10:
                specialty_sequences[spec].append(seq)
                doctor_sequences[doc_id].append(seq)

    os.makedirs("./results", exist_ok=True)

    # train & eval per specialty
    for specialty, sequences in specialty_sequences.items():
        print(f"\n Specialty: {specialty} | {len(sequences)} Sequences")

        # Tokenizer & Dataset
        token2idx = build_tokenizer(sequences)
        dataset = TreatmentSequenceDataset(sequences, token2idx)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # init
        model = TransformerAutoencoder(vocab_size=len(token2idx), max_len=30)
        model = train_model(model, dataloader, epochs=2)

        # eval reconstruction loss
        losses = evaluate(model, dataloader)
        print(f"Mean Reconstruction-Loss: {losses.mean():.4f}")

        # aggr losses per doctor
        doctor_scores = defaultdict(list)
        idx = 0
        for doc_id, seqs in doctor_sequences.items():
            if doctor_features[doc_id]['specialty'] != specialty:
                continue
            for _ in seqs:
                if idx < len(losses):
                    doctor_scores[doc_id].append(losses[idx])
                    idx += 1

        mean_scores = {
            doc: np.mean(scores) for doc, scores in doctor_scores.items() if scores
        }
        top_anomalies = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)[:1]

        print(f"\n Highest score for specialty ({specialty}):")
        for i, (doc, score) in enumerate(top_anomalies, 1):
            print(f"{i}. Doctor {doc} | Anomaly-Score: {score:.4f}")

        # ➤ SPEICHERE DIE SCORES ALS JSON
        with open(f"./results/transformer_scores_{specialty}.json", "w") as f:
            json.dump({str(doc): float(score) for doc, score in mean_scores.items()}, f, indent=2)

    # additional global training
    print("\n Training globally over all specialties...")
    all_sequences = []
    global_doctor_sequences = defaultdict(list)

    for doc_id, feat in doctor_features.items():
        for seq in feat.get('all_sequences', []):
            if len(seq) >= 2:
                all_sequences.append(seq)
                global_doctor_sequences[doc_id].append(seq)

    token2idx = build_tokenizer(all_sequences)
    dataset = TreatmentSequenceDataset(all_sequences, token2idx)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = TransformerAutoencoder(vocab_size=len(token2idx), max_len=30)
    model = train_model(model, dataloader, epochs=2)

    losses = evaluate(model, dataloader)
    print(f"Mean Reconstruction-Loss (global): {losses.mean():.4f}")

    # aggregation
    doctor_scores = defaultdict(list)
    idx = 0
    for doc_id, seqs in global_doctor_sequences.items():
        for _ in seqs:
            if idx < len(losses):
                doctor_scores[doc_id].append(losses[idx])
                idx += 1

    mean_scores = {
        doc: np.mean(scores) for doc, scores in doctor_scores.items() if scores
    }
    top_anomalies = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    print("\n Highest scores (global):")
    for i, (doc, score) in enumerate(top_anomalies, 1):
        print(f"{i}. Doctor {doc} | Anomaly-Score: {score:.4f}")

    # save
    with open("./results/transformer_scores_global.json", "w") as f:
        json.dump({str(doc): float(score) for doc, score in mean_scores.items()}, f, indent=2)


if __name__ == "__main__":
    main()
    print("This is where you pretend to be impressed")
