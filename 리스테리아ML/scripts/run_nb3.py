import sys, time, nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
sys.stdout.reconfigure(encoding='utf-8')
path = sys.argv[1]; cwd = sys.argv[2]
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=5400, kernel_name='python3', resources={'metadata': {'path': cwd}})
t0 = time.time()
try:
    client.execute()
    print(f'OK executed in {time.time()-t0:.0f}s')
except CellExecutionError as e:
    print(f'FAILED after {time.time()-t0:.0f}s')
    print(str(e)[:6000])
finally:
    nbformat.write(nb, path)
    print('saved', path)
