import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
import tempfile  # Added to handle file saving

# Load environment variables
load_dotenv()

# Load the GROQ and Google API keys
# groq_api_key = os.getenv('GROQ_API_KEY')
# 
groq_api_key = st.secrets["groq"]
google_api_key = st.secrets["google"]
os.environ["GOOGLE_API_KEY"] = google_api_key
# Streamlit app title
st.title("ASK YOUR DOC")

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

# Prompt template for answering questions based on document context
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

def vector_embedding(uploaded_file):
    # Check if vectors are already in session state
    if "vectors" not in st.session_state:
        # Ensure that a file has been uploaded
        if uploaded_file is not None:
            # Show a message while processing the document
            with st.spinner("Processing document and generating vector embeddings. Please wait..."):
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                # Load embeddings and documents
                st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                
                # Load PDF document using the temporary file path
                loader = PyPDFLoader(temp_file_path)
                st.session_state.docs = loader.load()

                # Split documents into chunks
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

                # Check if documents have been split correctly
                if st.session_state.final_documents:
                    # Generate vector embeddings
                    try:
                        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
                        st.success("Document processed successfully. You can now ask questions!")
                    except Exception as e:
                        st.error(f"Error creating FAISS vectors: {e}")
                else:
                    st.error("No documents to process for embeddings.")
        else:
            st.warning("Please upload a PDF document.")

# Upload PDF file and automatically process the document once uploaded
uploaded_file = st.file_uploader("Upload a PDF Document", type="pdf")

if uploaded_file:
    vector_embedding(uploaded_file)

# Text input for user's query (displayed after document is processed)
if "vectors" in st.session_state:
    st.subheader("Query the Document")
    prompt1 = st.text_input("Enter Your Question Related to the Document")

    # Handle query if document is uploaded and question is provided
    if prompt1:
        # Ensure embeddings have been generated before querying
        if "vectors" in st.session_state:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()

            # Create the retrieval chain
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Measure response time
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            st.write("Response time:", time.process_time() - start)

            # Display the response
            st.write(response['answer'])

            # Show similar documents in an expander
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response.get("context", [])):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        else:
            st.warning("Document processing is not complete yet.")
