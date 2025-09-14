import os
from pinecone import Pinecone,ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

if "rag-project-512" not in pc.list_indexes().names():
    pc.create_index(
        name="rag-project-512",
        dimension=1536,  
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index("rag-project-512")