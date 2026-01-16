import os
import boto3
import re
import shutil
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- CONFIGURATION ---
OCR_TEXT_PATH = "full_text_ocr.txt"
DB_PATH = "chroma_db"


def main():
    # Load the AWS profile from .env
    load_dotenv()
    os.environ["AWS_PROFILE"] = os.getenv("AWS_PROFILE")

    # Create Bedrock client
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )

    # Initialize embeddings model
    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id="amazon.titan-embed-text-v2:0"
    )

    print("🚀 Starting database loading process...")

    # 1. Load the OCR'd text
    if not os.path.exists(OCR_TEXT_PATH):
        print(f"❌ Error: Text file not found at '{OCR_TEXT_PATH}'")
        return

    print(f"📖 Loading text from '{OCR_TEXT_PATH}'...")
    with open(OCR_TEXT_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Parse sections with Regex (for municipal code format like "12.04.010")
    print("📑 Parsing text into sections using Regex...")
    section_pattern = r'(\d+\.\d+\.\d+)'
    splits = re.split(section_pattern, text)

    documents = []
    # Combine the section number with its content
    for i in range(1, len(splits), 2):
        if i + 1 < len(splits):
            section_number = splits[i]
            content = splits[i + 1]
            documents.append(
                Document(page_content=content.strip(), metadata={"section": section_number})
            )

    # 3. Fallback to chunking if Regex parsing is ineffective
    if len(documents) < 10:
        print("⚠️  Few sections found, using fallback chunking strategy...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        documents = text_splitter.create_documents([text])

    print(f"📄 Created {len(documents)} documents.")

    # 4. Clear out the old database
    if os.path.exists(DB_PATH):
        print("🗑️  Removing existing database...")
        shutil.rmtree(DB_PATH)

    print(f"🗄️  Initializing ChromaDB at '{DB_PATH}'...")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # 5. Add documents to the vector store
    print(f"⚡ Adding {len(documents)} documents to the database...")
    print("This will take a while. Go grab a coffee! ☕")
    db.add_documents(documents)
    print("✅ Documents added successfully.")

    # 6. Verify the database
    print("\n🔍 Verifying database...")
    try:
        collection_count = db._collection.count()
        print(f"✅ Database has {collection_count:,} documents!")

        # Run a test similarity search
        print("\nRunning a test search for 'fence height'...")
        test_results = db.similarity_search("fence height", k=3)

        if test_results:
            for doc in test_results:
                section = doc.metadata.get('section', 'Unknown')
                print(f"   📋 Result: Section {section} | {doc.page_content[:100]}...")
        else:
            print("❌ Test search returned no results.")
    except Exception as e:
        print(f"❌ Verification failed: {e}")

    print("\n🎉 COMPLETE! Database is ready.")


# Run the main function
if __name__ == "__main__":
    main()
