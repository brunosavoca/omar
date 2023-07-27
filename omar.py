import PyPDF2
import streamlit as st
from PyPDF2 import PdfReader 
from langchain. embeddings.openai import OpenAIEmbeddings 
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain. chains.question_answering import load_qa_chain 
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

# Set the API Key

load_dotenv() 
OPENAI_API_KEY='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'

API_KEY = os.environ['OPENAI_API_KEY']

# Set up the user interface layout
st.title("OmarGPT - Chateá con tus documentos")
pdf_file = st.file_uploader("Cargá tu PDF", type=["pdf"])

if pdf_file is not None:
    # Process the uploaded file
    reader = PdfReader(pdf_file)

    # read data from the file and put them into a variable called raw_text
    raw_text =''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
        
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()

    docsearch = FAISS.from_texts(texts, embeddings)

    #acá se puede utilizar diferentes modelos, en este caso será OpenAI
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    query = st.text_input("Qué necesitas saber de este documento?", "breve resumen de este documento")
    if st.button("Buscar"):
        docs = docsearch.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)
        st.write(answer)
