from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
import meilisearch
import asyncio
import json
import openai
import os
from termcolor import colored
import requests
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_secret(key):
    with open('secrets.json') as f:
        return json.load(f)[key]
    
openai.api_key = load_secret('openai-api-key')

encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

def custom_textwrap(text, width):
  def wrap_line(line):
    words = line.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
      if current_length + len(word) <= width:
        current_line.append(word)
        current_length += len(word) + 1
      else:
        lines.append(' '.join(current_line))
        current_line = [word]
        current_length = len(word) + 1

    lines.append(' '.join(current_line))
    return '\n'.join(lines)

  lines = text.split('\n')
  wrapped_lines = [wrap_line(line) for line in lines]
  return '\n'.join(wrapped_lines)

def custom_indent(text, prefix):
    lines = text.split('\n')
    return '\n'.join(prefix + line for line in lines)


def print_summary(hit, answer):
    title = f"\033[1m{hit['title']}\033[0m"
    id = f"\033[1m({hit['year']}/{create_hyperlinked_text(hit['id'])}):\033[0m"
    answer = custom_textwrap(answer, width=100)
    answer = custom_indent(answer, '   ')
    print(f"📝 {title} {id}\n{answer}")

def create_hyperlinked_text(text):
    import re

    def replace_with_link(match):
        match = match.group(0)
        return f'\033]8;;http://arxiv.org/abs/{match}\033\\{match}\033]8;;\033\\'

    return re.sub(r'\d{4}.\d{5}', replace_with_link, text)


async def search_quadrant(search_query):
  print(colored("🔎 Searching Qdrant...", 'cyan'))
  qdrant = QdrantClient()
  hits = qdrant.search(
    collection_name="papers",
    query_vector=encoder.encode(search_query).tolist(),
    limit=10
  )
  print(colored(f"✅ Done searching Qdrant ({len(hits)} results)", 'green'))
  return hits

async def search_meilisearch(search_query):
  print(colored(f"🔎 Searching MeiliSearch ('{search_query}')...", 'cyan'))

  client = meilisearch.Client('http://127.0.0.1:7700')
  index = client.index("papers")

  hits = index.search(search_query)
  
  print(colored(f"✅ Done searching MeiliSearch ({len(hits['hits'])} results)", 'green'))
  return hits

async def main():
  # Ask question
  search_query = input(colored("🧞‍♂️ Ask a question: ", 'cyan'))

  queries_string = simple_text_completion(
     f"""
Given the following question, create a good keyword query for searching academic titles and abstracts.
The keyword search ANDs together multiple keywords when they are about distinct topics.
                         
Example #1: What is alphafold?
Queries: alphafold

Example #2: What is the difference between a conformer and a transformer?
Queries: conformer, transformer
                         
Example #3: What are the state of the art techniques in protein structure prediction?
Queries: protein structure prediction state of the art

Example #4: how do you improve the performance of retrieval augmented generation?
Queries: retrieval augmented generation performance
                         
Question: {search_query}
Queries: 
  """)

  queries = queries_string.split(',')
  queries = [query.strip() for query in queries]

  print(colored(f"👀 Suggested queries: {queries}"))

  meilisearch_futures = []
  for query in queries:
     meilisearch_futures.append(search_meilisearch(query))
  qdrant_future = search_quadrant(search_query)

  qdrant_hits = await qdrant_future
  meilisearch_hits = []
  for future in meilisearch_futures:
     response = await future
     for hit in response['hits']:
      meilisearch_hits.append(hit)

  all_hits = qdrant_hits
  all_hits.extend(meilisearch_hits)

  # Normalize hits to contain just titles, IDs and abstracts.
  all_hits = [
    normalize_hit(hit)
    for hit in all_hits
  ]
  print(colored(f"📚 Total hits: {len(all_hits)}", 'yellow'))

  # Remove duplicates - where the title is the same between two entries.
  all_hits = [
    hit
    for hit in all_hits
    if hit['title'] not in [other_hit['title'] for other_hit in all_hits if other_hit != hit]
  ]

  print(colored(f"📚 Unique hits: {len(all_hits)}. Scoring with cross-encoder...", 'yellow'))

  model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

  scores = model.predict([
    (search_query, hit['abstract'])
    for hit in all_hits
  ])

  all_hits_with_scores = [
    (score, hit)
    for score, hit in sorted(zip(scores, all_hits), key=lambda x: x[0], reverse=True)
  ]

  all_hits = [
    hit
    for score, hit in all_hits_with_scores
    if score > 0
  ]

  dropped_hits = [
    hit
    for score, hit in all_hits_with_scores
    if score <= 0
  ]
  print(colored(f"🗑️  Dropped {len(dropped_hits)} hits:", 'grey'))
  for hit in dropped_hits:
    print(colored(f"  • {hit['title']}", 'grey'))

  print(colored(f"📚 Narrowed down to {len(all_hits)} hits.", 'yellow'))
  all_hits = all_hits[:10]
  print(colored(f"📚 Taking top {len(all_hits)} hits.", 'yellow'))

  from concurrent.futures import ProcessPoolExecutor

  summary_tasks = []
  with ProcessPoolExecutor(max_workers=10) as executor:
    for result in executor.map(compact_context_with_query, [(search_query, hit) for hit in all_hits]):
      (hit, answer) = result

      print_summary(hit, answer)
      print()

      summary = f'"{hit["title"]}" ({hit["year"]}/{hit["id"]}): {answer}'
      summary_tasks.append(summary)

  formatted_hits = "\n".join(summary_tasks)

  print(colored("🤖 Generating response (gpt-3.5-turbo)...", 'cyan'))
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "Answer the user's question with the information in context. If the answer is not in the context, respond with 'not in context' or use a function to look something up. For every sentence, produce a reference to the papers whose context you used in brackets along with the year it was last updated. For example: Dogs are kind [2023/1002.2515, 2009/1241.2502]. Consider that newer papers may have more cutting edge techniques. You must only answer from the context. Ignore irrelevant results in the context."},
      {"role": "user", "content": search_query},
      {"role": "user", "content": f"Context: {formatted_hits[:7000]}"},
    ],
    stream=True
  )

  print()
  print(colored("💬 Response:", 'blue'))

  for chunk in response:
    chunk_msg = chunk['choices'][0]['delta']
    content = chunk_msg.get('content')
    if content:
      print(colored(content, 'green'), end="", flush=True)
  print()

def compact_context_with_query(s):
  (query, hit) = s
  return compact_context_text(query, hit)
    
def normalize_hit(hit):
  return {
    'title': hit.payload.get('title'),
    'id': hit.payload.get('id', '').replace('-', '.'),
    'abstract': hit.payload.get('abstract'),
    'last_updated': hit.payload.get('update_date'),
    'year': hit.payload.get('update_date').split('-')[0],
  } if hasattr(hit, 'payload') else {
    'title': hit['title'],
    'id': hit['id'].replace('-', '.'),
    'abstract': hit['abstract'],
    'last_updated': hit.get('update_date'),
    'year': hit.get('update_date').split('-')[0],
  }

def compact_context_text(question, hit):
  instruction = f"""
> System: The user will give you a question and a passage. If any quotes from the passage help answer the question, respond with a summary of those passages. If no passages help answer the question, respond with [NONE].
> User:
Question: {question}
Passage: {hit['abstract']}
> Assistant:
Summary of relevant passages:
"""

  return (hit, simple_text_completion(instruction))

def simple_text_completion(prompt):
  """
  Does a simple completion with OpenAI's API using a plain HTTP call.
  """
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai.api_key}"
  }

  data = {
    "model": "text-davinci-003",
    "prompt": prompt,
    "max_tokens": 250,
    "temperature": 1,
    "top_p": 1,
    "stop": ["\n"]
  }

  response = requests.post(
    "https://api.openai.com/v1/completions",
    headers=headers,
    data=json.dumps(data)
  )

  response_json = response.json()

  return response_json["choices"][0]["text"]

def simple_chat_completion(instruction, context, model="gpt-3.5-turbo"):
  """
  Does a simple completion with OpenAI's API.
  """
  response = openai.ChatCompletion.create(
    messages=[
      {"role": "system", "content": instruction},
      {"role": "user", "content": context},
    ],
  )

  return response.choices[0]["message"]["content"]


asyncio.run(main())

