## importing generate function
from AugumentationAndGeneration import generate
## importing torch 
import torch
from dotenv import load_dotenv

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
import weaviate
import os
load_dotenv()
weaviate_url=os.getenv('WEAVIATE_URL')
weaviate_api_key=os.getenv('WEAVIATE_API_KEY')

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

def answer_question(
    query,
    context=None,
    top_k_semantic_and_keyword=50,
    top_k_from_combined_semantic_and_keyword=20,
    final_top_k=5,
    K=60,
    is_generate_summary=False
):
    """
    Processes one user question using the existing RAG pipeline.
    """

    if context is None:
        context = []

    # Add the current question to conversation history
    context.append({
        "role": "user",
        "content": query
    })

    # Generate answer using the existing RAG pipeline
    response = generate(
        query,
        model,
        tokenizer,
        model,
        tokenizer,
        embedding_model,
        context,
        collection,
        reranker,
        top_k_semantic_and_keyword=top_k_semantic_and_keyword,
        top_k_from_combined_semantic_and_keyword=top_k_from_combined_semantic_and_keyword,
        final_top_k=final_top_k,
        K=K,
        is_generate_summary=is_generate_summary
    )

    # Store the assistant response
    context.append({
        "role": "assistant",
        "content": response
    })

    return response
