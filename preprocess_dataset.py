import json
from typing import Generator

def get_dataset_generator(path: str) -> Generator:
    with open(path, "r") as fp:
        for line in fp:
            row = json.loads(line)
            yield row
        

def filter_generator(g: Generator, filter_fn):
    for item in g:
        if filter_fn(item):
            yield item

def stop_after(g, num_items):
    for i, item in enumerate(g):
        if i == num_items:
            break
        yield item

def clean_document(doc):
    return {
        "id": doc["id"],
        "title": doc["title"].replace("\n", " "),
        "abstract": doc["abstract"],
        "categories": doc["categories"].split(" "),
        "update_date": doc["update_date"],
    }

documents_list = []
try:
    with open("documents.json", "r") as fp:
        for line in fp:
            documents_list.append(json.loads(line))
except FileNotFoundError:
    dataset_generator = get_dataset_generator(
        path="arxiv-metadata-oai-snapshot.json"
    )

    def filter_relevant(doc):
        for category in doc["categories"]:
            if category.startswith("cs."):
                return True
        
        return False

    documents = map(clean_document, dataset_generator)
    documents = filter(filter_relevant, documents)

    print(f"Generating in-memory documents structure")
    documents_list = list(documents)

    print(f"Writing {len(documents_list)} documents...")
    with open("documents.json", "w") as fp:
        for doc in documents_list:
            fp.write(json.dumps(doc) + "\n")

print("Document examples:")
for doc in documents_list[:3]:
    print(f"[{doc['update_date']}] {doc['title']} ({doc['categories']})")

