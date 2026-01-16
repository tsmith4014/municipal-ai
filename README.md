# Municipal AI - RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, ChromaDB, and AWS Bedrock. This project demonstrates how to build an AI assistant that answers questions based on your own documents.

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/tsmith4014/municipal-ai.git
cd municipal-ai
```

### 2. Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure AWS Credentials

This project uses AWS Bedrock for embeddings and LLM. You need an AWS account with Bedrock access.

```bash
# Copy the template
cp .env_template .env

# Edit .env and set your AWS profile name
# AWS_PROFILE=your-profile-name
```

Make sure your AWS profile is configured in `~/.aws/credentials`.

### 4. Install System Dependencies

The PDF extraction requires Tesseract OCR:

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr
```

### 5. Add Your Source Document

Place your PDF in the `source_data/` directory:

```bash
mkdir -p source_data
# Copy your PDF to source_data/
```

Update `ingest.py` to point to your file:

```python
PDF_PATH = "source_data/your_document.pdf"
```

### 6. Run the Pipeline

```bash
# Step 1: Extract text from PDF
python ingest.py

# Step 2: Load text into vector database
python load_to_db.py

# Step 3: Start the chatbot
python main.py
```

---

## Project Structure

```
municipal-ai/
├── ingest.py          # PDF text extraction (Lab 1b)
├── load_to_db.py      # Vector database creation (Lab 2)
├── main.py            # RAG chatbot application (Lab 3)
├── check_db.py        # Database verification utility
├── requirements.txt   # Python dependencies
├── .env_template      # Environment config template
├── source_data/       # Place your PDFs here
├── chroma_db/         # Vector database (auto-generated)
└── full_text_ocr.txt  # Extracted text cache (auto-generated)
```

---

## Customizing for Your Own Data

### Example: D&D Rules Chatbot

Here's how to create a chatbot for Dungeons & Dragons rules using the free SRD:

#### 1. Download the D&D SRD PDF

```bash
curl -o source_data/dnd_srd.pdf "https://media.wizards.com/2016/downloads/DND/SRD-OGL_V5.1.pdf"
```

#### 2. Update `ingest.py`

```python
# Change this line:
PDF_PATH = "source_data/dnd_srd.pdf"
```

#### 3. Update the Prompt in `main.py`

```python
prompt_template = """
You are an expert Dungeon Master assistant for D&D 5th Edition. 
Your task is to answer questions based ONLY on the following context from the SRD.
If the context does not contain the answer, state that the information is not available.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
```

#### 4. Run the Pipeline

```bash
python ingest.py
python load_to_db.py
python main.py
```

Now ask questions like:
- "What spells can a level 3 wizard cast?"
- "How does grappling work?"
- "What are the stats for an Owlbear?"

---

## Advanced Customization

### Changing the Chunking Strategy

The default chunking in `load_to_db.py` uses regex to split by section numbers (like `12.04.010`). For different document types, modify the chunking logic:

#### Option 1: Change the Regex Pattern

```python
# In load_to_db.py, modify the section pattern:

# For documents with "Chapter X" sections:
section_pattern = r'(Chapter \d+)'

# For documents with "Section X.X" format:
section_pattern = r'(Section \d+\.\d+)'
```

#### Option 2: Adjust Chunk Size

For documents without clear sections, adjust the fallback chunker:

```python
# In load_to_db.py:
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,    # Increase for more context per chunk
    chunk_overlap=300,  # Increase for better continuity
)
```

#### Option 3: Custom Parsing Function

For structured data (like player profiles, product catalogs, etc.):

```python
def parse_custom_data(text):
    """Parse your specific document format."""
    documents = []
    
    # Example: Split by a custom pattern
    pattern = r'(ITEM \d+:.*?)(?=ITEM \d+:|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for i, content in enumerate(matches):
        documents.append(
            Document(
                page_content=content.strip(),
                metadata={"item_number": i + 1}
            )
        )
    
    return documents
```

### Changing the Embedding Model

To use a different AWS Bedrock embedding model:

```python
# In load_to_db.py and main.py:
embeddings = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v2:0"  # Change this
)
```

Available Bedrock embedding models:
- `amazon.titan-embed-text-v1`
- `amazon.titan-embed-text-v2:0`
- `cohere.embed-english-v3`

### Changing the LLM

To use a different AWS Bedrock LLM:

```python
# In main.py:
llm = ChatBedrock(
    model_id="us.amazon.nova-lite-v1:0",  # Change this
    client=bedrock_client,
    model_kwargs={
        "max_tokens": 1500,
        "temperature": 0.3  # Lower = more focused, Higher = more creative
    }
)
```

Available Bedrock LLMs:
- `us.amazon.nova-lite-v1:0`
- `us.amazon.nova-pro-v1:0`
- `anthropic.claude-3-sonnet-20240229-v1:0`
- `anthropic.claude-3-haiku-20240307-v1:0`

### Adjusting Retrieval

To retrieve more or fewer documents for context:

```python
# In main.py:
retriever = db.as_retriever(search_kwargs={'k': 5})  # Retrieve 5 docs instead of 3
```

---

## Troubleshooting

### "GOOGLE_API_KEY not found"
This project uses AWS Bedrock, not Google. Make sure your `.env` file has `AWS_PROFILE` set correctly.

### "No module named 'unstructured'"
Run `pip install -r requirements.txt` from your activated virtual environment.

### "Tesseract not found"
Install Tesseract OCR (see Step 4 above).

### Empty or poor results
- Check that `full_text_ocr.txt` contains your extracted text
- Try increasing `chunk_size` in `load_to_db.py`
- Try increasing `k` in the retriever

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   PDF File   │────▶│  ingest.py   │────▶│   Raw Text   │
└──────────────┘     │  (OCR/Parse) │     │   (.txt)     │
                     └──────────────┘     └──────┬───────┘
                                                 │
                                                 ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   ChromaDB   │◀────│ load_to_db.py│◀────│  Chunked     │
│   (Vectors)  │     │ (Embed/Store)│     │  Documents   │
└──────┬───────┘     └──────────────┘     └──────────────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   main.py    │────▶│   Bedrock    │────▶│   Answer     │
│  (Retrieve)  │     │    LLM       │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## License

MIT
