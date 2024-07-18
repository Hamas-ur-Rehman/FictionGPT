import tempfile
from flask import Flask, jsonify,redirect
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from langchain import VectorDBQA
from IPython.display import HTML, Audio, display
from base64 import b64decode
import scipy
from scipy.io.wavfile import read as wav_read
import io
import requests
import openai
import shutil
import gradio as gr
from helper import *
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import time

llm = ChatOpenAI(temperature=0)
embeddings = OpenAIEmbeddings()
def save_characters(query):
    purpose = """
    Your task is to identify characters from the given book and arrange them in a proper format as this:
    character1\n
    character2\n
    character3\n
    return back to the customer in that format only.
    FOLLOW THIS FORMAT ONLY DONOT SHARE ANYTHING EXCEPT CHARACTERS OF THE BOOK
    SHARE THE STORY CAHARACTERS
    You will be given the name of the book
    """

    messages = [SystemMessage(content=purpose),
        HumanMessage(content=query)]
    result = llm(messages)
    with open(f'{destination_folder}/{query}.txt', 'a') as file:
        file.write(result.content)
    return f'{result.content}'





characters_folder = f'./characters/'
destination_folder = './characters'
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

with open('books.txt', 'a') as f:
    pass

def fetch_books():
    with open(f'books.txt', 'r') as file:
        lines = [line.strip() for line in file if line.strip()]
        books_list = sorted(list(set(lines)))
    return books_list
        
def fetch_characters():
    book_chracters = {}
    books_list = fetch_books()
    for i in books_list:
        with open(f'{characters_folder}{i}.txt', 'r') as file:
            lines = [line.strip() for line in file if line.strip()]
            lines_value = sorted(list(set(lines)))
            book_chracters[i] = lines_value
    return book_chracters

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)



with gr.Blocks(title="FictionGPT") as demo:
        gr.Markdown(
            """
        <h1 align="center">FictionGPT 🎅</h1>
        <h6 align="center">✨ Characters Brought to life ✨</h6>
            """
        )
        
        with gr.Tab("ChatBot"):
            with gr.Row():
                    books = gr.Dropdown(label="Book")
                    fetch_book = gr.Button(value="Fetch Books")
            with gr.Row():
                    book_character = gr.Dropdown(label="Book Character")
                    book_btn = gr.Button(value="Fetch Selected Book Characters")
            with gr.Row(variant="panel"):
                text = gr.inputs.Audio(source='microphone',type="filepath", label="Upload Audio File")
            with gr.Column(variant="panel"):
                    speaker = gr.Dropdown(label="Voice", choices=speaker_name,visible=True)
            with gr.Row(variant="panel"):
                generate = gr.Button("Generate")
            with gr.Row(variant="panel"):
                audio = gr.Audio()

            def fetch_everything():
                books_list = fetch_books()
                books_list = sorted(list(set(books_list)))
                value = books_list[0]
                return gr.Dropdown.update(choices=books_list,value=value, visible=True,interactive=True
                    )
            def value(book: str):
            
                books_list = fetch_books()
                        
                book_chracters = fetch_characters()
                c = book_chracters[book]
                value = c[0]
                return gr.Dropdown.update(choices=c, value=value, visible=True,interactive=True
                    )
        with gr.Tab("Upload PDF"):
            
            book_label = gr.Text(placeholder="Book Name")
            with gr.Row():                
                    upload = gr.inputs.File(label="Upload your book",file_count="multiple")
            upload_btn = gr.Button(value="upload")
            error_markdown = gr.Markdown()

        book_btn.click(value,inputs=[books],outputs=book_character)
        fetch_book.click(fetch_everything,outputs=books)
        
        def process_file(file,book_label,progress=gr.Progress()):
            progress(0, desc="Starting")
            
            print(file)
            if type(file) is list:
                if book_label=="" or book_label is None:
                    with open(f'books.txt', 'r') as file:
                        lines = [line.strip() for line in file if line.strip()]
                        books_list = sorted(list(set(lines)))
                    return gr.Markdown.update("""
                        <img style="float: left;" src="https://cdn-icons-png.flaticon.com/512/463/463612.png " width="20" height="20">
                        &nbsp
                        Error Occured : Please add Valid Book Name
                        """),gr.Dropdown.update(choices=books_list, value=books_list[0], visible=True,interactive=True)
            
                book_name = fix_collection_name(book_label.lower())
                persist_directory = f'db/{book_name}'
                docsearch = Chroma(collection_name=book_name, persist_directory=persist_directory, embedding_function=embeddings)
                docsearch.persist()
                progress(0.05,desc="Loading Documents to database please be paitient this takes time (it is not stuck!)")
                
                for i in file:
                    if check_pdf(i.name):
                        pdf_folder_path = i.name
                        loader = UnstructuredPDFLoader(pdf_folder_path, strategy="fast")
                        documents = loader.load()
                        texts = text_splitter.split_documents(documents)
                        Chroma.add_texts(docsearch,texts=[t.page_content for t in texts])
                        print(f'Loaded Document: {i.name}')
                        
                        
                    else:
                        return gr.Markdown.update("""
                        <img style="float: left;" src="https://cdn-icons-png.flaticon.com/512/463/463612.png " width="20" height="20">
                        &nbsp
                        Error Occured : Only PDF files accepted as valid input
                        """),gr.Dropdown.update(choices=books_list, value=books_list[0], visible=True,interactive=True)
                    progress(0.5)
                    docsearch = None
                    progress(0.7,desc="Fetching book characters")
                    
                    save_characters(book_label)
                    
                    
                    with open(f'books.txt', 'a') as f:
                        f.write(f'{book_label}\n')
                        
                    with open(f'books.txt', 'r') as file:
                        lines = [line.strip() for line in file if line.strip()]
                        books_list = sorted(list(set(lines)))
                    
                    
                    progress(0.9, desc="Saving Book")
                        
                    time.sleep(5)

                            
                    return gr.Markdown.update("""
                    <img style="float: left;" src="https://cdn-icons-png.flaticon.com/512/845/845646.png " width="20" height="20">
                    &nbsp
                    Success : Your Files have been processed you can try them out in the chatbot tab
                    """),gr.Dropdown.update(choices=books_list, value=books_list[0], visible=True,interactive=True)
                        
            else:
                return gr.Markdown.update("""
                <img style="float: left;" src="https://cdn-icons-png.flaticon.com/512/463/463612.png " width="20" height="20">
                &nbsp
                Error Occured : Please select a PDF to upload
                """),gr.Dropdown.update(choices=books_list, value=books_list[0], visible=True,interactive=True)   
                
            
        upload_btn.click(process_file,inputs=[upload,book_label],outputs=[error_markdown,books])
        
        
        def synthesize_audio(audio_path: str, speaker_str: str = "",book_character: str = "", books:str = ""):
            collection_name = fix_collection_name(books.lower())
            persist_directory = f'db/{collection_name}'
            docsearch = Chroma(collection_name=collection_name,persist_directory=persist_directory,embedding_function=embeddings)
            voice_data = find_voice_by_name(id,speaker_str)
            if not audio_path:
                return None
            print(audio_path)
            save_audio(audio_path)
            with open('./transctibe_audio.wav', "rb") as f:
                transcript = openai.Audio.transcribe("whisper-1", f)
                response = chatbot(transcript['text'],voice_data['voice_id'],book_character,books,docsearch)
                return gr.Audio.update(response)
        generate.click(synthesize_audio, inputs=[text, speaker,book_character,books], outputs=audio)


a,b,url = demo.launch(
    # server_name="0.0.0.0",
    enable_queue=True,
    # ssl_verify=False,
    favicon_path="favicon.png",
    share = True,
    prevent_thread_lock=True,
    # quiet = True
)

app = Flask(__name__)
CHECK = False
@app.route('/', methods=['GET'])
def index():
    # global CHECK
    # global url
    # global text
    # if not CHECK:
    #     fp = tempfile.TemporaryFile()
    #     fp.write(url.encode('utf-8'))
    #     fp.seek(0)
    #     text = fp.read()
    #     text = text.decode('utf-8')
    #     print(text)
    return redirect(location=url)


if __name__ == '__main__':
    app.run(debug=True)
