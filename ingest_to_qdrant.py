

from qdrant_client import models, QdrantClient
import hashlib
from concurrent.futures import ProcessPoolExecutor
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def upload_records_process(documents_chunk):
    qdrant = QdrantClient()

    qdrant.upload_records("papers", [
        models.Record(
            id=hashlib.md5(doc["id"].encode()).hexdigest(),
            vector=doc["vector"],
            payload=doc
        ) for doc in documents_chunk
    ])


print("Loading encoder...")
encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

print(f"Opening documents file...")

documents_list = []
with open("documents.json", "r") as fp:
    for line in fp:
        documents_list.append(json.loads(line))

print(f"Indexing {len(documents_list)} documents...")

batch_size = 4096
documents_list_chunked = [documents_list[i:i + batch_size] for i in range(0, len(documents_list), batch_size)]

qdrant = QdrantClient()
qdrant.recreate_collection(
    collection_name="papers",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
        distance=models.Distance.COSINE
    )
)

# We want to upload the documents in parallel with continuing
# to encode the next batch of documents. If we don't do this,
# then we have a lot of GPU idle time while docs are being
# uploaded to Qdrant.
upload_executor = ProcessPoolExecutor(max_workers=3)

for documents_chunk in tqdm(documents_list_chunked, desc="Processing document chunks"):
    abstracts = encoder.encode([doc["abstract"] for doc in documents_chunk])
    for idx, doc in enumerate(documents_chunk):
        doc["vector"] = abstracts[idx].tolist()

    upload_executor.submit(upload_records_process, documents_chunk)

# Wait for the executors to finish
upload_executor.shutdown()
