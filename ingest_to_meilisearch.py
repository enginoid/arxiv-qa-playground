"""
This module reads all the docs and ingests them into MeiliSearch.
"""
import json
import meilisearch
from tqdm import tqdm

def yield_docs():
    with open("documents.json", "r") as fp:
        for line in fp:
            yield json.loads(line)

docs = list(yield_docs())

client = meilisearch.Client('http://127.0.0.1:7700')

index = client.index("papers")

# Replace any . in the doc ID with a - becasue MeiliSearch doesn't like dots
for doc in docs:
    doc["id"] = doc["id"].replace(".", "-")

batch_size = 100
chunked_docs = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]

for doc_chunk in tqdm(chunked_docs, desc="Indexing documents"):
    index.add_documents(doc_chunk, primary_key="id")
