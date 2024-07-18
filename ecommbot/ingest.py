from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv
import os
from ecommbot.data_converter import data_converter
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")

# Initialize SentenceTransformer model
# Initialize HuggingFace embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def ingest_data(status):
    vstore = AstraDBVectorStore(
        embedding=embedding,
        collection_name="chatbotecomm",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_KEYSPACE,
    )
    
    if status is None:
        docs = data_converter()  # Assuming data_converter is a function defined in data_converter.py
        inserted_ids = vstore.add_documents(docs)
        return vstore, inserted_ids  # Return both vstore and inserted_ids if data was ingested
    else:
        return vstore  # Return vstore if status is not None

if __name__=='__main__':
    vstore, inserted_ids = ingest_data(None)
    print(f"\nInserted {len(inserted_ids)} documents.")
    
    # Perform a similarity search
    query_text = "can you tell me the low budget sound basshead."
    results = vstore.similarity_search(query_text)
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")
