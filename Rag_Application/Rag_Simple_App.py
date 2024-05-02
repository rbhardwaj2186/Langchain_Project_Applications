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

PATH = 'D:/Work\Gre/UTD/Courses/Spring_II/Exams/Tensorflow_developer/Python_3.9/Langchains/pythonProject1/Rag_Application/Churchill_speech.pdf'
#document_path = "D:/Work\Gre/UTD/Courses/Spring_II/Exams/Tensorflow_developer/Python_3.9/Langchains/pythonProject1/Rag_Application/Churchill_speech.pdf"
document_path = PATH
#loader = Docx2txtLoader(document_path)
loader = PyPDFLoader(document_path)
documents = loader.load()

# SEGMENTING THE DOCUMENT INTO SEGMENTS

text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# USING CHROMADB FOR DOCUMENT EMBEDDING

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

# RETRIEVAL CHAINS
# STUFF DOCUMENT CHAIN

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

query = "What is the Document about?"
print(qa.invoke(query))

'''Output: {'query': 'What is the Document about?', 'result': " The document is about the British government's efforts to defend against potential invasion 
 Nazi Germany, including their military strategies and measures taken to prevent enemy activity within the country. It also discusses the recent military losses and successes in the war, including the evacuation of British and French troops from the beaches of Dunkirk."}'''

# MAP-REDUCE DOCUMENT CHAIN

#qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="map_reduce", retriever=docsearch.as_retriever())

#query = "what are the wonders of earth ?"
#qa.run(query)

# REFINE DOCUMENT CHAIN
#qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="refine", retriever=docsearch.as_retriever())

#query = "what are the wonders of earth ?"
#qa.invoke(query)
