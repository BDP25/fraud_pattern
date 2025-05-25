```mermaid
---
config:
  layout: dagre
---
flowchart TB
 subgraph subGraph0["Data Layer"]
        DB[("SQL Database<br>Multi-GB Medical Records")]
        PQ["Parquet Files<br>Columnar Format"]
  end
 subgraph subGraph1["Feature Extraction Pipeline (feature_extractor.py)"]
        DS["DoctorDataset<br>IterableDataset"]
        FE["DoctorFeatureExtractor<br>Chunked Processing"]
        CACHE[("Feature Cache<br>doctor_features.pkl")]
  end
 subgraph subGraph2["Feature Engineering"]
        AGG["Doctor Aggregation<br>• Billing patterns<br>• Treatment frequencies<br>• Sequence extraction"]
        META["Metadata Processing<br>• Invoice types<br>• Tariff information<br>• Patient context"]
        SEQ["Sequence Generation<br>• Chronological treatments<br>• Patient-level sequences<br>• Specialty grouping"]
  end
 subgraph subGraph3["ML Training Pipeline (transformer_model.py)"]
        LOAD["Feature Loading"]
        SPEC["Specialty Grouping<br>Cardiology, Dermatology, etc."]
        TOK["Tokenization<br>Treatment codes → indices"]
        DS2["TreatmentSequenceDataset<br>Metadata integration"]
  end
 subgraph subGraph4["Model Architecture"]
        TRANS["CTXTransformer<br>Autoencoder"]
        EMB["Embedding Layer<br>+ Positional Encoding"]
        MLP["Metadata MLP<br>Categorical + Continuous"]
        ATTN["Multi-Head Attention<br>Sequence Modeling"]
        DEC["Decoder<br>Reconstruction Loss"]
  end
 subgraph subGraph5["Training & Evaluation"]
        TRAIN["Per-Specialty Training<br>Adam Optimizer<br>CrossEntropy Loss"]
        EVAL["Reconstruction Error<br>Per-sequence scoring"]
        AGG2["Doctor-level Aggregation<br>Mean anomaly scores"]
  end
 subgraph subGraph6["Output & Results"]
        JSON["JSON Results<br>transformer_scores_*.json"]
        RANK["Anomaly Ranking<br>Top diverging doctors"]
  end
 subgraph subGraph8["Configuration & Control"]
        CONFIG["Configuration<br>• Column mappings<br>• Chunk sizes<br>• Test/Full modes"]
  end
    DB --> PQ
    PQ --> DS
    DS --> FE
    FE --> CACHE & AGG & META & SEQ
    CACHE --> LOAD
    LOAD --> SPEC
    SPEC --> TOK
    TOK --> DS2
    DS2 --> TRANS
    TRANS --> EMB & MLP & TRAIN
    EMB --> ATTN
    MLP --> ATTN
    ATTN --> DEC
    TRAIN --> EVAL
    EVAL --> AGG2
    AGG2 --> JSON
    JSON --> RANK
    CONFIG -.-> DS & FE & LOAD
     DB:::datastore
     PQ:::datastore
     DS:::processing
     FE:::processing
     CACHE:::datastore
     AGG:::processing
     META:::processing
     SEQ:::processing
     LOAD:::processing
     SPEC:::processing
     TOK:::processing
     DS2:::processing
     TRANS:::ml
     EMB:::ml
     MLP:::ml
     ATTN:::ml
     DEC:::ml
     TRAIN:::ml
     EVAL:::ml
     AGG2:::ml
     JSON:::output
     RANK:::output
     CONFIG:::config
    classDef datastore fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef ml fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef output fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef config fill:#fce4ec,stroke:#880e4f,stroke-width:2px
