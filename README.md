# PDF ChatBot

A Streamlit-based PDF chatbot that uses LangChain, FAISS, and Hugging Face Transformers for document-based question answering. The app allows users to upload a PDF file, process it, and ask questions based on the contents of the PDF.

## Features
- Upload a PDF file and extract its contents.
- Split the text into chunks for efficient document retrieval.
- Generate embeddings using Hugging Face embeddings and store them in FAISS for fast retrieval.
- Use a fine-tuned language model (e.g., Flan-T5 or any other model) to answer questions based on the document.
- Chat interface to interact with the PDF content.
- Supports seamless integration with Hugging Face Hub and local models.

## Requirements

To run this application, make sure you have the following Python packages installed:

```bash
pip install -r requirements.txt

