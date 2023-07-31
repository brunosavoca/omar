import streamlit as st
import openai
import requests
from bs4 import BeautifulSoup
import PyPDF2
from io import BytesIO

# Esto es para el sidebar, donde el usuario ingresa la clave de la API de OpenAI
openai.api_key = st.sidebar.text_input("Introduce tu OpenAI API Key")

# Interacción con PDFs
st.header('Interactuar con PDFs')

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
            if query:
                docs = docsearch.similarity_search(query)
                answer = chain.run(input_documents=docs, question=query)
                st.write(answer)
