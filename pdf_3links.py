import streamlit as st
import openai
import requests
from bs4 import BeautifulSoup
import PyPDF2
from io import BytesIO

# Esto es para el sidebar, donde el usuario ingresa la clave de la API de OpenAI
openai.api_key = st.sidebar.text_input("Introduce tu OpenAI API Key")

# Aquí es donde el usuario ingresará las URLs
url1 = st.text_input("Introduce la URL 1")
url2 = st.text_input("Introduce la URL 2")
url3 = st.text_input("Introduce la URL 3")

def get_content_from_url(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    return soup.text

def generate_article(url1, url2, url3):
    content1 = get_content_from_url(url1)
    content2 = get_content_from_url(url2)
    content3 = get_content_from_url(url3)

    combined_content = content1 + " " + content2 + " " + content3

    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=combined_content,
      temperature=0.5,
      max_tokens=1000
    )
    
    return response.choices[0].text.strip()

if st.button('Generar artículo'):
    if url1 and url2 and url3 and openai.api_key:
        article = generate_article(url1, url2, url3)
        st.write(article)
    else:
        st.write("Por favor, asegúrate de haber ingresado todas las URLs y tu OpenAI API Key.")

# Interacción con PDFs
st.header('Interactuar con PDFs')

uploaded_file = st.file_uploader("Elige un archivo PDF", type="pdf")
if uploaded_file is not None:
    pdf_file = PyPDF2.PdfFileReader(BytesIO(uploaded_file.getvalue()))
    num_pages = pdf_file.numPages
    text_from_pdf = ''
    for page in range(num_pages):
        text_from_pdf += pdf_file.getPage(page).extractText()

    question = st.text_input("Introduce tu pregunta")

    if st.button('Obtener respuesta del PDF'):
        if question and openai.api_key and text_from_pdf:
            response = openai.Completion.create(
              engine="text-davinci-002",
              prompt=f'{text_from_pdf}\nQuestion: {question}\nAnswer:',
              temperature=0.5,
              max_tokens=100
            )
            st.write(response.choices[0].text.strip())
        else:
            st.write("Por favor, asegúrate de haber cargado el PDF, ingresado una pregunta y tu OpenAI API Key.")
