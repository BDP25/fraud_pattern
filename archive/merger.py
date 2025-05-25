import dask.dataframe as dd

# Load datasets (no memory explosion)
invoice_lines = dd.read_parquet('./data/invoice_lines.parquet')
invoices_2022 = dd.read_parquet('./data/invoices_2022.parquet')
lerb = dd.read_parquet('./data/lerb.parquet')

# Merge without triggering compute
merged = invoice_lines.merge(invoices_2022, on='invoice_id', how='left')
merged = merged.merge(lerb, left_on='INVH_AccountId', right_on='acc_id', how='left')

# Save to Parquet (partitioned)
merged.to_parquet('./data/merged_dask_output', engine='pyarrow')
