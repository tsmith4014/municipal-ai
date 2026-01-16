import boto3
from dotenv import load_dotenv
import os
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
DB_PATH = "chroma_db"


def main():
    load_dotenv()
    os.environ["AWS_PROFILE"] = os.getenv("AWS_PROFILE")

    print("Initializing AI Assistant...")

    # Create Bedrock client
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )

    # Initialize embeddings (must match what was used in load_to_db.py)
    embeddings = BedrockEmbeddings(
        client=bedrock_client,
        model_id="amazon.titan-embed-text-v2:0"
    )

    # Connect to ChromaDB
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={'k': 3})

    # --- Quick Test of the Retriever ---
    print("\n--- Testing the retriever ---")
    question = "What is the rule for fence height?"
    retrieved_docs = retriever.invoke(question)
    print(f"Retriever found {len(retrieved_docs)} documents.")
    if retrieved_docs:
        print("Top result preview:")
        print(retrieved_docs[0].page_content[:400])
    print("-" * 25)

    # --- Create the Prompt Template ---
    prompt_template = """
You are an expert assistant on municipal codes. Your task is to answer questions based ONLY on the following context.
If the context does not contain the answer, state that the information is not available in the provided documents.
Do not use any outside knowledge.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    prompt = PromptTemplate.from_template(prompt_template)

    # --- Initialize the LLM (AWS Bedrock) ---
    llm = ChatBedrock(
        model_id="us.amazon.nova-lite-v1:0",
        client=bedrock_client,
        model_kwargs={
            "max_tokens": 1500,
            "temperature": 0.3
        }
    )

    # --- Build the RAG Chain ---
    # This chain: prompt -> llm -> parse output to string
    answer_chain = prompt | llm | StrOutputParser()

    # Complete chain that returns BOTH the answer AND source documents
    # RunnableParallel runs retriever and passthrough simultaneously
    # .assign() adds the LLM answer to the output dictionary
    rag_chain = (
        RunnableParallel(
            context=retriever,
            question=RunnablePassthrough()
        ).assign(answer=answer_chain)
    )

    print("\nAI Assistant is ready. Ask a question or type 'exit' to quit.")

    # --- Interactive Q&A Loop ---
    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() == 'exit':
            break

        response = rag_chain.invoke(user_question)

        # Print the sources
        print("\n--- Sources ---")
        for i, doc in enumerate(response["context"]):
            section = doc.metadata.get("section", "N/A")
            print(f"{i+1}. Section: {section}")
            print(f"   Content: {doc.page_content[:200]}...")

        # Print the answer
        print("\nAssistant's Answer:")
        print(response["answer"])


if __name__ == "__main__":
    main()
