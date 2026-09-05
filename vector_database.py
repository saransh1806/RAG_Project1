## importing necessary dependencies
from pathlib import Path
import torch
device="cuda" if torch.cuda.is_available() else "cpu"
## Choosing embedding model for vectorizing chunks
embedding_model_name="BAAI/bge-small-en-v1.5"
from sentence_transformers import SentenceTransformer
## Getting embedding model using hugging face library "sentence transformer"
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5",device=device)

## Setting weaviate for vector database
import weaviate
weaviate_url = "x054kyyoq1mg2wlmquknq.c0.eu-central-1.aws.weaviate.cloud"
weaviate_api_key = "cDB5WjZpR25IekRCaGJCNV9YQzI4ZUNlUi9QblZFTUtTSXVDNEhEVU1GS1FmY01rVURvNU9FaDlFcXRjPV92MjAw"

# Connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=weaviate_api_key,
)

## Using langchain pdf loader for loading our dataset
from langchain_community.document_loaders import PyPDFLoader
DATA_PATH = Path("data")

documents = []

for pdf_file in DATA_PATH.glob("*.pdf"):
    ## pdf loader
    loader = PyPDFLoader(str(pdf_file))
    docs = loader.load()

    # Add filename to metadata
    for doc in docs:
        doc.metadata["source_file"] = pdf_file.name

    documents.extend(docs)

print("\nTotal pages loaded:", len(documents))

## Chunking documents
## Chunking documents using Recursive
## It tries to split text using a prioritized list of separators—defaulting to paragraphs ("\n\n"), sentences ("\n"), words (" "), and characters ("")
from langchain_text_splitters import RecursiveCharacterTextSplitter
## text_splitter object using chunk_size=1000 i.e max chunk size could be 1000 and a overlap of size 150 to preserve context during splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i
    chunk.metadata["course"] = "MIT 6.006"
## Getting text langchain documents(chunks)
texts = [chunk.page_content for chunk in chunks]
## Converting textss into embedding using previously loaded embedding_model
embeddings = embedding_model.encode(
    texts,
    normalize_embeddings=True
)

from weaviate.classes.config import Configure, Property, DataType

collection_name = "MIT6006Chunk"
if client.collections.exists(collection_name):
       client.collections.delete(collection_name)
collection = client.collections.create(
                    name=collection_name,
    
                    vector_config=Configure.Vectors.self_provided(),
    
                    properties=[
                        Property(
                            name="text",
                            data_type=DataType.TEXT
                        ),
    
                        Property(
                            name="source_file",
                            data_type=DataType.TEXT
                        ),
    
                        Property(
                            name="page",
                            data_type=DataType.INT
                        ),
    
                        Property(
                            name="chunk_id",
                            data_type=DataType.INT
                        ),
    
                        Property(
                            name="course",
                            data_type=DataType.TEXT
                        )
                    ]
                )


## Adding chunks into collection with  property listed below
with collection.batch.fixed_size(batch_size=100) as batch:

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

        batch.add_object(
            properties={
                "text": chunk.page_content,
                "source_file": chunk.metadata["source_file"],
                "page": chunk.metadata["page"],
                "chunk_id": chunk.metadata["chunk_id"],
                "course": chunk.metadata["course"]
            },
            vector=embedding.tolist()
        )

print('done')
