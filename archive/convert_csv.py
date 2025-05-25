import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm

def convert_csv_to_parquet(input_csv, output_parquet):
    """
    Convert a CSV file to Parquet format.

    Parameters:
    input_csv (str): Path to the input CSV file.
    output_parquet (str): Path to the output Parquet file.
    """
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv, delimiter=';', low_memory=False)

    # Convert the DataFrame to a PyArrow Table
    table = pa.Table.from_pandas(df)

    # Write the table to a Parquet file
    pq.write_table(table, output_parquet)

if __name__ == "__main__":

    files = ['pos_codes.csv',
             'invoices_2022.csv',
             'lerb.csv',
             'invoice_lines.csv']
    
    for file in tqdm(files):
        input_csv = f'./data/{file}'
        output_parquet = f'./data/{file.split(".")[0]}.parquet'
        convert_csv_to_parquet(input_csv, output_parquet)
        print(f"Converted {input_csv} to {output_parquet}")
