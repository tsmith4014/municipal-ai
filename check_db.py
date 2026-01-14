import os
import chromadb
import boto3
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURATION ---
DB_PATH = "chroma_db"

def check_with_direct_client():
    """Uses the chromadb client to connect and inspect the database."""
    print("\n--- Method 1: Direct ChromaDB Client Check ---")

    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database directory not found at '{DB_PATH}'")
        return

    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collections = client.list_collections()

        if not collections:
            print("❌ No collections found in the database.")
            return

        print(f"✅ Found {len(collections)} collections:")
        for collection in collections:
            count = collection.count()
            print(f"   - Collection '{collection.name}': {count:,} documents")

    except Exception as e:
        print(f"❌ An error occurred with the direct client: {e}")


def check_with_langchain_wrapper():
    print("\n--- Method 2: LangChain Wrapper Check ---")

    try:
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )

        embeddings = BedrockEmbeddings(
            client=bedrock_client,
            model_id="amazon.titan-embed-text-v2:0"
        )

        db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )

        print("\nRunning a test search for 'selling ice'...")
        test_results = db.similarity_search("selling ice", k=3)

        for i, doc in enumerate(test_results):
            section = doc.metadata.get("section", "Unknown")
            print(f"   {i+1}. Section {section}: {doc.page_content[:150]}...")

    except Exception as e:
        print(f"❌ An error occurred with the LangChain wrapper: {e}")


if __name__ == "__main__":
    print("🚀 Running Comprehensive Database Check...")
    check_with_direct_client()
    check_with_langchain_wrapper()
    print("\n✅ Database check complete.")