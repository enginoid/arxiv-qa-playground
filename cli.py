import subprocess

command = """\
import meilisearch
meilisearch_client = meilisearch.Client('http://127.0.0.1:7700')
meilisearch_index = meilisearch_client.index("papers")
"""

try:
    subprocess.run(["ipython", "-i", "-c", command])
except FileNotFoundError:
    print("IPython is not installed. Please install it by running: pip install ipython")
