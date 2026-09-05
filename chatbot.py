from AugumentationAndGeneration import generate
import torch
device="cuda" if torch.cuda.is_available() else "cpu"

from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
embedding_model_name="BAAI/bge-small-en-v1.5"
from sentence_transformers import SentenceTransformer
## Getting embedding model using hugging face library "sentence transformer"
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5",device=device)



from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

model_name="Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)

import weaviate
import os
weaviate_url = "x054kyyoq1mg2wlmquknq.c0.eu-central-1.aws.weaviate.cloud"
weaviate_api_key = "cDB5WjZpR25IekRCaGJCNV9YQzI4ZUNlUi9QblZFTUtTSXVDNEhEVU1GS1FmY01rVURvNU9FaDlFcXRjPV92MjAw"

# Connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=weaviate_api_key,
)
collection = client.collections.get("MIT6006Chunk")

from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def print_response(response):
    # ANSI escape codes
    BOLD = "\033[1m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    RESET = "\033[0m"
    RED = "\033[31m"

    if response['role'] == 'assistant':
        color = GREEN
    elif response['role'] == 'user':
        color = BLUE
    else:
        color=RED

    s = f"{BOLD}{color}{response['role'].capitalize()}{RESET}: {response['content']}"
    print(s)

def chat(context,
        model=model,
        tokenizer=tokenizer,
            hyde_model=model,
            hyde_tokenizer=tokenizer,
            embedding_model=embedding_model,
            reranker=reranker,
             collection=collection,
             top_k_semantic_and_keyword=50,
             top_k_from_combined_semantic_and_keyword=20,
             final_top_k=5,
             K=60,
             is_generate_summary=False
         ):
    
    # Start by printing the initial assistant prompt
    print_response(context[-1])
    
    # Continues until the user types 'STOP'
    while True:
        query = input()
        if query.lower() == 'stop':
            break

        # Generate the response based on the user's prompt and existing context
        response = generate(query,
             model,
             tokenizer,
             hyde_model,
             hyde_tokenizer,
             embedding_model,
             context,
             collection,
            reranker,
             top_k_semantic_and_keyword=50,
             top_k_from_combined_semantic_and_keyword=20,
             final_top_k=5,
             K=60,
             is_generate_summary=is_generate_summary)
        # Append the user's prompt and the assistant's response to the context
        context.append({"role": "user", "content": query})
        context.append({"role": "assistant", "content": response})

        # Print the most recent user output, followed by the assistant response
        print_response(context[-2])
        print_response(context[-1])
    client.close()

