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
            response = openai.Completion.create(
              engine="gpt-3.5-turbo",
              prompt=f'{text_from_pdf}\nQuestion: {question}\nAnswer:',
              temperature=0.5,
              max_tokens=5000
            )
            st.write(response.choices[0].text.strip())
        else:
            st.write("Por favor, asegúrate de haber cargado el PDF, ingresado una pregunta y tu OpenAI API Key.")
