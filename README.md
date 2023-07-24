# Arxiv QA

Retrieval-augmented generation example that answers questions from Arxiv abstracts and titles.

![arxiv-retrieval-anns](https://github.com/enginoid/arxiv-qa-playground/assets/62200/48365997-2157-4a35-9796-235abfb47abb)
(Video sped up 3x.)

## Setup

* Copy `secrets-example.json` and replace with your own key.
* Fetch `arxiv-metadata-oai-snapshot.json`
  * `kaggle datasets download -d Cornell-University/arxiv`
* Run `preprocess_dataset.py`
   * Input file: `arxiv-metadata-oai-snapshot.json`
   * Output file: `documents.json` (a bit smaller)
* `docker compose up -d` to run MeiliSearch and Qdrant
* Then
    * `ingest_to_meilisearch.py`
    * `ingest_to_qdrant.py`
        * You'll want a GPU 😁, use `nvitop` to check it's using GPU.
        * Example performance: g5.xlarge (1x A10G), ~600k abstracts, ~12 minutes
* Finally `query.py` to ask some questions.

# Other tips

* You can connect to a nice server to test Meilisearch keyword lookup on `http://localhost:8080/`
* `cli.py` could be useful but at the moment only exposes `meilisearch_index` and `meilisearch_client`
