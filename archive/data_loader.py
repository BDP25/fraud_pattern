import os
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset

class ParquetDataset(IterableDataset):
    def __init__(self, folder_path):
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.parquet')]

    def __iter__(self):
        for file in self.files:
            table = pq.read_table(file)
            df = table.to_pandas()
            for _, row in df.iterrows():
                yield self._convert_to_tensor(row)

    def _convert_to_tensor(self, row):
        # Customize: convert row to tensor(s)
        return row.values  # Simplified

    def __len__(self):
        total_length = 0
        for file in self.files:
            table = pq.read_table(file)
            total_length += table.num_rows
        return total_length

class ParquetDataLoader:
    def __init__(self, folder_path, batch_size=32):
        self.dataset = ParquetDataset(folder_path)
        self.batch_size = batch_size
        self.iterator = iter(self.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            try:
                batch.append(next(self.iterator))
            except StopIteration:
                if not batch:
                    raise
                break
        return batch

# Example usage
if __name__ == "__main__":
    folder_path = '.\data\merged_dask_output'
    data_loader = ParquetDataLoader(folder_path, batch_size=10)

    for batch in data_loader:
        print(batch)  # Process your batch here