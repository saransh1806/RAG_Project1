## importing necessary dependencies
from pathlib import Path
import torch

## setting device agnostic code
device="cuda" if torch.cuda.is_available() else "cpu"

## Choosing embedding model for vectorizing chunks and query
embedding_model_name="BAAI/bge-small-en-v1.5"
from sentence_transformers import SentenceTransformer
## Getting embedding model using hugging face library "sentence transformer"
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5",device=device)

## Setting weaviate for vector database
import weaviate
from dotenv import load_dotenv
load_dotenv()
import os
weaviate_url=os.environ['weaviate_url']
weaviate_api_key=os.environ['weaviate_api_key']

# Connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=weaviate_api_key,
)

## Using langchain pdf loader for loading our dataset
from langchain_community.document_loaders import PyPDFLoader
DATA_PATH = Path("data")

documents = []
## loading every pdf of dataset using langchain "PyPDFLoader".Converting each page of pdf into langchain document,adding pdf filename as metadata and storing them all into one list(documents)
for pdf_file in DATA_PATH.glob("*.pdf"):
    ## pdf loader
    loader = PyPDFLoader(str(pdf_file))
    ## it get each page from the given pdf_file and automatically puts page information into the document's metadata.
    ## docs is a Python list of LangChain Document objects
    docs = loader.load() 

    # Add filename to metadata
    for doc in docs:
        doc.metadata["source_file"] = pdf_file.name ## putting source_file name as metadata 

    documents.extend(docs)

## looking at how many langchain document are there i.e no of pdf pages in dataset
print("\nTotal pages loaded:", len(documents))

## Chunking documents using Recursive Splitting
## It tries to split text using a prioritized list of separators—defaulting to paragraphs ("\n\n"), sentences ("\n"), words (" "), and characters ("")
from langchain_text_splitters import RecursiveCharacterTextSplitter
## text_splitter object using chunk_size=1000 i.e max chunk size could be 1000 and a overlap of size 150 to preserve context during splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
## It splits documents into chunks using langchain object text splitter defined above
chunks = text_splitter.split_documents(documents)
for i, chunk in enumerate(chunks):
    ## setting metadata like chunk_id and course for easy retrieval ahead 
    chunk.metadata["chunk_id"] = i
    chunk.metadata["course"] = "MIT 6.006"
## Extracting text content from each chunk
texts = [chunk.page_content for chunk in chunks]
## Converting texts into embedding using previously loaded embedding_model to create vector database
embeddings = embedding_model.encode(
    texts,
    normalize_embeddings=True
)

from weaviate.classes.config import Configure, Property, DataType
## creating a collection to store embeddings using vector database weaaviate
collection_name = "MIT6006Chunk"
## If collection already exist delete the previous one and create a fresh new collection since weaviate allow only 1 collection for free tier
## or we delete the previous collection and create fresh new collection if there is some changes in our dataset like we are adding or removing any files
if client.collections.exists(collection_name):
       client.collections.delete(collection_name)

## metadata/schema that each object in my collection will have.
## Each object in this schema is a chunk

 ##Object
# ├── vector
# │   └── [0.021, -0.183, 0.492, ...]
# │
# ├── text
# │   └── "Explain binary search in short"
# │
# ├── source_file
# │   └── "mit_cse.pdf"
# │
# ├── page
# │   └── 42
# │
# ├── chunk_id
# │   └── 157
# │
# └── course
#     └── "CSE"

collection = client.collections.create(
                    name=collection_name,
    
                    vector_config=Configure.Vectors.self_provided(),## we have our predefined embedding model,that's why vector_config is self provided
                    ## Setting properties for weaviate collection
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


## Adding chunks/object into collection with  property listed below
with collection.batch.fixed_size(batch_size=100) as batch:## adding 1 batch i.e 100 chunks at a time

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

        batch.add_object(
            properties={
                "text": chunk.page_content,## gtting text from given chunk
                ## adding source_file,page_no,chunk_id and course_name as metadata for keeping track of all the objects later
                "source_file": chunk.metadata["source_file"],
                "page": chunk.metadata["page"],
                "chunk_id": chunk.metadata["chunk_id"],
                "course": chunk.metadata["course"]
            },
            vector=embedding.tolist()
        )

print('done')

### looking at overall structure

## We have 131 pdf divided into around 1800 pages then converted into chunk using recusivetextsplitter.
## We created a schema to store embeddings and metadata of chunks using embedding model and weaviate vector database.
## Each chunk is an object stored in weaviate collections having properties associated with them is (text,source_file,page,chunk_id and course) and a embedding_vector


### overall pipeline

# 131 PDFs
#    │
#    ▼
# ~1800 PDF pages
#    │
#    │ PyPDFLoader
#    ▼
# 1800-ish Documents
#    │
#    │ RecursiveCharacterTextSplitter
#    ▼
# Chunks
#    │
#    ├───────────────┐
#    │               │
#    ▼               ▼
# Metadata        Embedding Model
#    │               │
#    │               ▼
#    │          Embedding Vector
#    │               │
#    └───────┬───────┘
#            ▼
#        Weaviate
#            │
#            ▼
#       Collection
#            │
#      ┌─────┴─────┐..........................................................................
#      ▼           ▼
#    Object      Object
#    Chunk 1     Chunk 2
#      │           │
#      ▼           ▼
#   properties  properties
#   + vector    + vector
