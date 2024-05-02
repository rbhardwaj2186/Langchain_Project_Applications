from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage, AIMessage
import os

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# Function to load document
def load_document(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())
    return qa

# Streamlit app
st.title('Document Retrieval Chat')

uploaded_file = st.file_uploader("Upload PDF document", type=['pdf'])

if uploaded_file is not None:
    st.write('Document successfully loaded.')
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    qa = load_document("temp.pdf")
    st.session_state.messages = []  # Clear previous messages
else:
    st.write('Please upload a PDF document.')

user_prompt = st.text_input(label='Send a message')

if user_prompt:
    st.session_state.messages.append(
        HumanMessage(content=user_prompt)
    )
    with st.spinner('Working on your request ... '):
        response = qa.invoke(user_prompt)
    st.text("Response:")
    st.write(response['result'])