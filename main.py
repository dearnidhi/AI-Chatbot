import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile
import os

# Load environment variables from .env file
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not huggingface_token:
    st.error("HUGGINGFACEHUB_API_TOKEN is not set in the .env file.")
else:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_token

# Set up Streamlit page
st.set_page_config(page_title="PDF ChatBot", page_icon="📚")
st.title("📚 PDF ChatBot")

# Initialize session state for storing QA chain
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Function to process the uploaded PDF
def process_pdf(uploaded_file):
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load and split the PDF into text chunks
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Generate embeddings and store them in a FAISS vector store
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(texts, embeddings)

    # Clean up the temporary file
    os.unlink(tmp_file_path)

    return db

# Sidebar for PDF upload
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if st.sidebar.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            db = process_pdf(uploaded_file)
            llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.5, "max_length": 512})

            # Define the prompt template for the QA system
            template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            Use three sentences maximum and keep the answer concise. 
            {context}
            Question: {question}
            Helpful Answer:"""
            QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

            # Create the QA chain
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=db.as_retriever(),
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
        st.sidebar.success("PDF processed successfully!")

# Main chat interface
st.header("Chat with your PDF")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your PDF"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.qa_chain is not None:
        # Get the response from the QA chain
        response = st.session_state.qa_chain.run(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("Please upload and process a PDF first.")

# Sidebar usage instructions
st.sidebar.markdown("""
## How to use:
1. Upload a PDF file (max 3MB)
2. Click 'Process PDF'
3. Ask questions about the PDF content
""")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit, LangChain, and Hugging Face")
