## importing generate function
from AugumentationAndGeneration import generate
## importing torch 
import torch

## setting device agnostic code
device="cuda" if torch.cuda.is_available() else "cpu"

from transformers import BitsAndBytesConfig
## Quantizing our model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, ## loading our model into 4 bit  
    bnb_4bit_compute_dtype=torch.float16
)
## using BAAI as embedding model
embedding_model_name="BAAI/bge-small-en-v1.5"
from sentence_transformers import SentenceTransformer
## Getting pretrained embedding model using hugging face library "sentence transformer"
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5",device=device)

## importing Autotokenizer and AutoModel
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

model_name="Qwen/Qwen2.5-7B-Instruct"
## getting pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
## getting pretrained model
## we will use this model for generating text and creating hypothetical document which will save space of loading two different model or two instance of the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config ## it quantizes our model and makes it efficient for loading 
)
import os
from dotenv import load_dotenv
load_dotenv()

import weaviate
weaviate_url=os.environ['weaviate_url']
weaviate_api_key=os.environ['weaviate_api_key']

# Connecting to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=weaviate_api_key,
)
## getting previously made collection
collection = client.collections.get("MIT6006Chunk")

## importing CrossEncoder to get reranker 
from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

## this function makes chatbot look visually better 
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

## This function makes an active chatbot where user gives it query and our generate function
## generates answer based on the query, print query and answer and finally  append the query
## and generated text into context.The chatbot finally ends when query.lower()=='stop'.
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
             top_k_semantic_and_keyword=top_k_semantic_and_keyword,
             top_k_from_combined_semantic_and_keyword=top_k_from_combined_semantic_and_keyword,
             final_top_k=final_top_k,
             K=K,
             is_generate_summary=is_generate_summary)
        # Append the user's prompt and the assistant's response to the context
        context.append({"role": "user", "content": query})
        context.append({"role": "assistant", "content": response})

        # Print the most recent user output, followed by the assistant response
        print_response(context[-2])
        print_response(context[-1])
    client.close()

