import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import json

class TreatmentSequenceWithMetadataDataset(Dataset):
    def __init__(self, data, token2idx, metadata_info, max_len=30):
        """
        Args:
            data: List of dicts with keys 'sequence' and 'metadata'
            token2idx: Dict mapping treatment codes to indices
            metadata_info: Dict mapping metadata fields to type and (optionally) normalization info
            max_len: Max sequence length (excl. CTX token)
        """
        self.token2idx = token2idx
        self.max_len = max_len
        self.metadata_info = metadata_info
        self.data = data

        self.preprocess_metadata()

    def preprocess_metadata(self):
        """
        Compute normalization stats for continuous metadata fields.
        """
        # Aggregate values
        self.metadata_stats = {}
        for key, info in self.metadata_info.items():
            if info['type'] == 'continuous':
                values = [d['metadata'][key] for d in self.data if key in d['metadata']]
                if values:  # Only compute stats if we have values
                    mean = np.mean(values)
                    std = np.std(values)
                    self.metadata_stats[key] = {'mean': mean, 'std': std}
                else:
                    self.metadata_stats[key] = {'mean': 0.0, 'std': 1.0}

    def encode_sequence(self, seq):
        tokens = [self.token2idx.get(t, 0) for t in seq]
        tokens = tokens[:self.max_len]  # truncate
        tokens = [0] + tokens  # prepend dummy for [CTX] token
        pad_len = self.max_len + 1 - len(tokens)
        tokens += [0] * pad_len  # pad
        return tokens

    def encode_metadata(self, meta):
        encoded = []
        for key, info in self.metadata_info.items():
            val = meta.get(key, None)

            if info['type'] == 'categorical':
                idx = info['vocab'].get(val, 0)  # fallback to 0 if unknown
                encoded.append(idx)

            elif info['type'] == 'continuous':
                if val is not None:
                    stats = self.metadata_stats[key]
                    norm_val = (val - stats['mean']) / (stats['std'] + 1e-8)
                    encoded.append(norm_val)
                else:
                    encoded.append(0.0)  # Default for missing values

        return torch.tensor(encoded, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        input_ids = self.encode_sequence(sample['sequence'])
        metadata = self.encode_metadata(sample['metadata'])

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'metadata': metadata
        }


def build_tokenizer(sequences):
    vocab = set()
    for seq in sequences:
        cleaned = [t for t in seq if isinstance(t, str) and t.strip()]
        vocab.update(cleaned)
    token2idx = {token: idx + 1 for idx, token in enumerate(sorted(vocab))}
    token2idx['<PAD>'] = 0
    return token2idx


def prepare_dataset_from_features(doctor_features, metadata_keys):
    """
    Convert doctor_features format to the format expected by the dataset.
    """
    data = []
    
    print(f"Processing {len(doctor_features)} doctors...")
    processed = 0
    
    for doc_id, features in doctor_features.items():
        processed += 1
        if processed % 100 == 0:
            print(f"  Processed {processed}/{len(doctor_features)} doctors")
            
        # Get sequence_metadata for this doctor
        sequence_metadata = features.get('sequence_metadata', [])
        
        # Each item in sequence_metadata already has both sequence and metadata
        for item in sequence_metadata:
            if isinstance(item, dict) and 'sequence' in item:
                seq = item['sequence']
                if len(seq) >= 2:  # Only include sequences with at least 2 items
                    meta = item.get('metadata', {})
                    
                    data.append({
                        'sequence': seq,
                        'metadata': meta,
                        'doctor_id': doc_id  # Keep track for evaluation
                    })
    
    print(f"Created {len(data)} data samples total")
    return data


def create_metadata_info(data, metadata_keys):
    """
    Analyze the metadata to create metadata_info structure.
    """
    metadata_info = {}
    
    for key in metadata_keys:
        # Collect all values for this key
        values = []
        for item in data:
            if key in item['metadata'] and item['metadata'][key] is not None:
                values.append(item['metadata'][key])
        
        if not values:
            continue
            
        # Determine if categorical or continuous
        if isinstance(values[0], str) or len(set(values)) < 20:
            # Categorical
            unique_vals = sorted(set(values))
            vocab = {val: idx + 1 for idx, val in enumerate(unique_vals)}
            vocab[None] = 0  # For missing values
            
            metadata_info[key] = {
                'type': 'categorical',
                'vocab': vocab
            }
        else:
            # Continuous
            metadata_info[key] = {
                'type': 'continuous'
            }
    
    return metadata_info


class CTXTransformerAutoencoder(nn.Module):
    def __init__(self, vocab_size, metadata_dim, d_model=128, nhead=4, num_layers=4, max_len=30):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len + 1, d_model))  # +1 for CTX token

        # MLP to project metadata to d_model
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, x, metadata):
        emb = self.embedding(x) + self.pos_encoder[:, :x.size(1)]

        # metadata shape: [B, d_model] -> [B, 1, d_model] -> broadcast along sequence length
        meta_emb = self.metadata_encoder(metadata).unsqueeze(1)
        meta_emb = meta_emb.expand(-1, x.size(1), -1)

        # inject metadata into embeddings
        emb = emb + meta_emb

        out = self.transformer(emb, emb)
        logits = self.decoder(out)
        return logits


def train_model(model, dataloader, epochs=5, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = batch['input_ids'].to(device)        # Fixed: use 'input_ids' not 'sequence'
            metadata = batch['metadata'].to(device)
            y = x.clone()

            logits = model(x, metadata)

            # reshape logits and target for loss: [B*T, V] and [B*T]
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

    return model


def evaluate(model, dataloader):
    device = next(model.parameters()).device
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    seq_losses = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['input_ids'].to(device)
            metadata = batch['metadata'].to(device)
            y = x.clone()

            logits = model(x, metadata)

            # loss: [B*T]
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            # reshape: [B, T] then average across sequence length
            loss = loss.view(x.size(0), x.size(1))
            seq_loss = loss.mean(dim=1).cpu().numpy()  # per-sequence loss
            seq_losses.extend(seq_loss)

    return np.array(seq_losses)


def main():
    # AEK: Configuration toggles
    TEST_MODE = False  # AEK: Set to True for testing with limited data, False for full processing
    MAX_SPECIALTIES = 3 if TEST_MODE else None  # AEK: Limit specialties in test mode
    MAX_SAMPLES_PER_SPECIALTY = 1000 if TEST_MODE else None  # AEK: Limit samples in test mode
    
    feature_file = "./data/doctor_features.pkl"
    assert os.path.exists(feature_file), "please run feature_extractor.py first."

    # Load extracted doctor features with sequence_metadata
    with open(feature_file, 'rb') as f:
        doctor_features = pickle.load(f)

    print(f"Loaded features for {len(doctor_features)} doctors")
    print(f"Running in {'TEST' if TEST_MODE else 'FULL'} mode")

    # Group all sequences by specialty and doctor (used only for evaluation aggregation)
    print("Grouping sequences by specialty...")
    specialty_sequences = defaultdict(list)
    doctor_sequences = defaultdict(list)
    
    processed_docs = 0
    for doc_id, feat in doctor_features.items():
        processed_docs += 1
        if processed_docs % 1000 == 0:
            print(f"Processed {processed_docs}/{len(doctor_features)} doctors")
            
        spec = feat.get('specialty', 'unknown')
        sequence_metadata = feat.get('sequence_metadata', [])
        
        # Extract sequences from sequence_metadata
        for item in sequence_metadata:
            if isinstance(item, dict) and 'sequence' in item:
                seq = item['sequence']
                if len(seq) >= 2:
                    specialty_sequences[spec].append(seq)
                    doctor_sequences[doc_id].append(seq)

    print(f"Found {len(specialty_sequences)} specialties:")
    for spec, seqs in specialty_sequences.items():
        print(f"  {spec}: {len(seqs)} sequences")

    os.makedirs("./results", exist_ok=True)

    # Train per specialty
    specialty_items = list(specialty_sequences.items())
    specialty_items.sort(key=lambda x: len(x[1]), reverse=True)  # Sort by number of sequences
    
    # Apply specialty limit if in test mode
    if MAX_SPECIALTIES:
        specialty_items = specialty_items[:MAX_SPECIALTIES]
        print(f"\nProcessing top {MAX_SPECIALTIES} specialties (out of {len(specialty_sequences)} total) - TEST MODE")
    else:
        print(f"\nProcessing all {len(specialty_items)} specialties - FULL MODE")
    
    for specialty, sequences in specialty_items:
        print(f"\nProcessing specialty: {specialty} | {len(sequences)} sequences")

        # Filter doctors for current specialty
        print("Filtering doctors for specialty...")
        doctor_ids = [
            doc_id for doc_id, feat in doctor_features.items()
            if feat.get('specialty', 'unknown') == specialty
        ]
        print(f"Found {len(doctor_ids)} doctors for specialty {specialty}")
        
        filtered_features = {doc_id: doctor_features[doc_id] for doc_id in doctor_ids}

        # Prepare dataset in correct format
        print("Preparing dataset...")
        metadata_keys = ['INVH_InvoiceTypeId', 'INVL_TariffType', 'INVL_Amount']
        dataset_data = prepare_dataset_from_features(filtered_features, metadata_keys)
        
        print(f"Prepared {len(dataset_data)} data samples")
        
        if not dataset_data:
            print(f"No valid data for specialty {specialty}, skipping...")
            continue

        # Limit dataset size if in test mode
        if MAX_SAMPLES_PER_SPECIALTY and len(dataset_data) > MAX_SAMPLES_PER_SPECIALTY:
            print(f"Limiting dataset to {MAX_SAMPLES_PER_SPECIALTY} samples (from {len(dataset_data)}) - TEST MODE")
            dataset_data = dataset_data[:MAX_SAMPLES_PER_SPECIALTY]

        # Create metadata info
        print("Creating metadata info...")
        metadata_info = create_metadata_info(dataset_data, metadata_keys)
        
        print(f"Metadata info created: {list(metadata_info.keys())}")
        
        if not metadata_info:
            print(f"No valid metadata for specialty {specialty}, skipping...")
            continue

        # Tokenizer and dataset
        token2idx = build_tokenizer(sequences)

        dataset = TreatmentSequenceWithMetadataDataset(
            data=dataset_data,
            token2idx=token2idx,
            metadata_info=metadata_info,
            max_len=30
        )
        
        if len(dataset) == 0:
            print(f"Empty dataset for specialty {specialty}, skipping...")
            continue
            
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Initialize and train the model
        metadata_dim = dataset[0]['metadata'].shape[0]
        model = CTXTransformerAutoencoder(
            vocab_size=len(token2idx),
            metadata_dim=metadata_dim,
            max_len=30
        )
        model = train_model(model, dataloader, epochs=5)

        # Evaluate reconstruction error
        losses = evaluate(model, dataloader)
        print(f"Mean reconstruction error: {losses.mean():.4f}")

        # Aggregate error scores by doctor
        doctor_scores = defaultdict(list)
        idx = 0
        for item in dataset_data:
            if idx < len(losses):
                doctor_scores[item['doctor_id']].append(losses[idx])
                idx += 1

        mean_scores = {
            doc: np.mean(scores) for doc, scores in doctor_scores.items() if scores
        }
        
        if not mean_scores:
            print(f"No scores computed for specialty {specialty}")
            continue
            
        top_anomalies = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        print(f"\nTop 5 diverging doctors ({specialty}):")
        for i, (doc, score) in enumerate(top_anomalies, 1):
            print(f"{i}. Doctor {doc} | anomaly-score: {score:.4f}")

        # Save scores to JSON
        with open(f"./results/transformer_scores_{specialty}.json", "w") as f:
            json.dump({str(doc): float(score) for doc, score in mean_scores.items()}, f, indent=2)


if __name__ == "__main__":
    main()