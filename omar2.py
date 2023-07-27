import PyPDF2
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

# Set the API Key (replace "YOUR_OPENAI_API_KEY" with your actual API key)
openai_api_key = "OPENAI_API_KEY"

# Set up the user interface layout
st.title("OmarGPT - Chateá con tus documentos")
pdf_file = st.file_uploader("Cargá tu PDF", type=["pdf"])

if pdf_file is not None and pdf_file.size > 0:
    # Process the uploaded file
    reader = PdfReader(pdf_file)

    # read data from the file and put them into a variable called raw_text
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    docsearch = FAISS.from_texts(texts, embeddings)  # Make sure FAISS is available

    #acá se puede utilizar diferentes modelos, en este caso será OpenAI
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    query = st.text_input("Qué necesitas saber de este documento?", "breve resumen de este documento")
    if st.button("Buscar"):
        docs = docsearch.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)
        st.write(answer)
else:
    st.write("Subí un archivo PDF válido.")