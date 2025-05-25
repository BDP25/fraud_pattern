import os
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset
from collections import Counter, defaultdict
import pickle
from tqdm import tqdm

class DoctorDataset(IterableDataset):
    """
    Dataset that reads parquet files and yields doctor-specific data
    """
    def __init__(self, folder_path, columns=None, chunk_size=1000000): # AEK: adjust chunk size according to memory
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.parquet')]
        self.columns = columns  
        self.chunk_size = chunk_size 
        
        
    def __iter__(self):
        doctor_buffer = defaultdict(list)
        
        for file in self.files:
            parquet_file = pq.ParquetFile(file)
            # read chunks
            for batch in tqdm(parquet_file.iter_batches(batch_size=self.chunk_size, 
                                                        columns=self.columns),
                                                        desc=f"Reading {os.path.basename(file)}"):
                batch_df = batch.to_pandas()
                
                for doctor_id in batch_df['INVH_AccountId'].unique():
                    doctor_data = batch_df[batch_df['INVH_AccountId'] == doctor_id]
                    yield doctor_id, doctor_data

class DoctorFeatureExtractor:
    """
    Extracts features from doctor data for anomaly detection
    """
    def __init__(self, column_mappings=None, metadata_columns=None):
        """
        Initialize with column mappings to handle different column names
        
        Args:
            column_mappings: Dictionary mapping standard names to actual column names
                             e.g., {'treatment_code': 'procedure_code'}
        """
        # AEK: adjust default column mappings
        self.columns = {
            'doctor_id': 'INVH_AccountId',
            'patient_id': 'INVH_ClientId',
            'treatment_code': 'INVL_Code',
            'amount': 'INVL_Amount',
            'date': 'INVL_DateBegin',
            'specialty_code': 'ACC_SpecialityCode',
            'specialty_name': 'SPE_Description'
        }
        self.metadata_columns = metadata_columns or []
        # update custom mappings
        if column_mappings:
            self.columns.update(column_mappings)
        
        # store features
        self.doctor_features = {}
    
    def _extract_metadata_for_patient(self, patient_df):
        meta = {}
        for col in self.metadata_columns:
            values = patient_df[col].dropna().unique()
            if len(values) == 1:
                meta[col] = values[0]
            elif len(values) > 1:
                meta[col] = values[0]  # Arbitrarily pick first
            else:
                meta[col] = None
        return meta
        
    def extract_features(self, doctor_id, doctor_df):
        """
        Extract features for a single doctor
        
        Args:
            doctor_id: ID of the doctor
            doctor_df: DataFrame containing this doctor's data
        
        Returns:
            Dictionary of features for this doctor
        """
        # Map column names to standardized names
        df = doctor_df.rename(columns={v: k for k, v in self.columns.items() if v in doctor_df.columns})

        # Extract specialty - fix the column mapping issue
        specialty = None
        if 'specialty_code' in df.columns and not df['specialty_code'].isna().all():
            specialty = df['specialty_code'].iloc[0]
        elif 'specialty_name' in df.columns and not df['specialty_name'].isna().all():
            specialty = df['specialty_name'].iloc[0]
        
        # average billing per treatment type
        avg_billing = df.groupby('treatment_code')['amount'].mean().to_dict()
        
        # treatment frequencies
        treatment_counts = df['treatment_code'].value_counts().to_dict()
        total_treatments = sum(treatment_counts.values())
        treatment_freqs = {k: v/total_treatments for k, v in treatment_counts.items()}
        
        # calculate typical sequences, treatment pairs
        sequences = []
        for patient_id, patient_df in df.groupby('patient_id'):
            treatments = patient_df.sort_values('date')['treatment_code'].tolist()
            for i in range(len(treatments)-1):
                sequences.append((treatments[i], treatments[i+1]))
        
        seq_counter = Counter(sequences)
        top_sequences = dict(seq_counter.most_common(5))
        
        # extract sequences and metadata
        sequence_metadata_pairs = []
        for patient_id, patient_df in df.groupby('patient_id'):
            treatments = patient_df.sort_values('date')['treatment_code'].tolist()
            if len(treatments) > 1:
                metadata = self._extract_metadata_for_patient(patient_df)
                sequence_metadata_pairs.append({
                    "sequence": treatments,
                    "metadata": metadata
                })
        
        # store features
        features = {
            'specialty': specialty,
            'avg_billing': avg_billing,
            'treatment_freqs': treatment_freqs,
            'top_sequences': top_sequences,
            'sequence_metadata': sequence_metadata_pairs,
            #'all_sequences': all_sequences,
            'patient_count': df['patient_id'].nunique(),
            'avg_treatments_per_patient': len(df) / df['patient_id'].nunique(),
            'total_amount': df['amount'].sum()
        }
        
        self.doctor_features[doctor_id] = features
        
        return features
    
    def process_dataset(self, dataset, max_doctors=None):
        """
        Process the entire dataset of doctors
        
        Args:
            dataset: DoctorDataset instance
            max_doctors: Optional maximum number of doctors to process
        
        Returns:
            Dictionary of all doctor features
        """
        processed_count = 0
        
        for doctor_id, doctor_df in tqdm(dataset, desc="Processing sequences"):
            self.extract_features(doctor_id, doctor_df)
            
            processed_count += 1
            if max_doctors and processed_count >= max_doctors:
                break
        
        return self.doctor_features

    def prepare_sequence_data(self, min_seq_length=2):
        """
        Prepare treatment sequence data for transformer model by specialty
        
        Returns:
            Dictionary mapping specialties to lists of sequences
        """
        specialty_sequences = defaultdict(list)
        doctor_sequences = defaultdict(list)
        
        # Group sequences
        for doctor_id, features in self.doctor_features.items():
                specialty = features.get('specialty', 'unknown')
                for item in features.get('sequence_metadata', []):
                    if len(item["sequence"]) >= min_seq_length:
                        specialty_sequences[specialty].append(item)
                        doctor_sequences[doctor_id].append(item)

        return specialty_sequences, doctor_sequences
    
    def save_features(self, path="./data/doctor_features.pkl"):
        with open(path, 'wb') as f:
            pickle.dump(self.doctor_features, f)
        print(f"Features saved as: {path}")

    def load_features(self, path="./data/doctor_features.pkl"):
        with open(path, 'rb') as f:
            self.doctor_features = pickle.load(f)
        print(f"Features loaded from: {path}")


def main():
    folder_path = './data/merged_dask_output' # AEK: extend to SQL or dump data to path
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} does not exist. Please check the path.")
    
    columns = ['INVH_AccountId', # AEK: specify columns as needed
               'INVH_ClientId', 
               'INVL_Code', 
               'INVL_Amount', 
               'INVL_DateBegin', 
               'ACC_SpecialityCode']
    
    metadata_columns = ['INVH_InvoiceTypeId', # AEK: specify metadata for patients
                        'INVL_TariffType']
    
    all_columns = list(set(columns + metadata_columns))

    features_file = "./data/doctor_features.pkl" # AEK: save features where desired
    
    extractor = DoctorFeatureExtractor(metadata_columns=metadata_columns)
    
    if os.path.exists(features_file):
        extractor.load_features(features_file)
        doctor_features = extractor.doctor_features
        print("Doctor features loaded from existing file.")
    else:
        dataset = DoctorDataset(folder_path, columns=all_columns)
        print("Extracting doctor features...")
        doctor_features = extractor.process_dataset(dataset)
        extractor.save_features(features_file)
    
    # Prepare sequence data for transformer model
    print("\nPreparing sequence data for transformer model...")
    specialty_sequences, doctor_sequences = extractor.prepare_sequence_data()
    
    print(f"Collected sequences for {len(specialty_sequences)} specialties")
    for specialty, sequences in specialty_sequences.items(): # AEK: sanity check, disable if desired
        print(f"  {specialty}: {len(sequences)} sequences")

if __name__ == "__main__":
    main()