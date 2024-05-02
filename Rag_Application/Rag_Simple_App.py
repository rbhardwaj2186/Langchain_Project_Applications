from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
import os

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# LOADING THE DOCUMENT WITH LOADER

document_path = "D:/Work\Gre/UTD/Courses/Spring_II/Exams/Tensorflow_developer/Python_3.9/Langchains/pythonProject1/Rag_Application/Churchill_speech.pdf"
#loader = Docx2txtLoader(document_path)
loader = PyPDFLoader(document_path)
documents = loader.load()

# SEGMENTING THE DOCUMENT INTO SEGMENTS

text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# USING CHROMADB FOR DOCUMENT EMBEDDING