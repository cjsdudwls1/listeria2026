import sys, pyarrow.parquet as pq
for p in sys.argv[1:]:
    f=pq.ParquetFile(p); print('##',p.split('리스테리아')[-1], f.metadata.num_rows, 'rows'); print(', '.join(f.schema_arrow.names))
