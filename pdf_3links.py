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

uploaded_file = st.file_uploader("Elige un archivo PDF", type="pdf")
if uploaded_file is not None:
    pdf_file = PyPDF2.PdfReader(BytesIO(uploaded_file.getvalue()))
    num_pages = len(pdf_file.pages)
    text_from_pdf = ''
    for page in range(num_pages):
        text_from_pdf += pdf_file.pages[page].extract_text()

    question = st.text_input("Introduce tu pregunta")

    if st.button('Obtener respuesta del PDF'):
        if question and openai.api_key and text_from_pdf:
            response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=[
                {"role": "system", "content": text_from_pdf},
                {"role": "user", "content": question},
              ]
            )
            st.write(response['choices'][0]['message']['content'])
        else:
            st.write("Por favor, asegúrate de haber cargado el PDF, ingresado una pregunta y tu OpenAI API Key.")
