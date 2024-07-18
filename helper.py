from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain import VectorDBQA
from scipy.io.wavfile import read as wav_read
from IPython.display import HTML, Audio, display
from base64 import b64decode
from dotenv import load_dotenv
import os
import requests
import re
import scipy
import io
import requests
import openai
import shutil

load_dotenv()
def check_pdf(string):
    return string.endswith('.pdf')

def fix_collection_name(collection_name):
    # Replace spaces with underscores
    collection_name = collection_name.replace(' ', '_')

    # Add underscore placeholders if the name is too short
    if len(collection_name) < 3:
        collection_name += '_' * (3 - len(collection_name))

    # Cut off the name if it is too long
    if len(collection_name) > 63:
        collection_name = collection_name[:63]

    # Replace consecutive periods with a single period
    collection_name = re.sub(r'\.\.', '.', collection_name)

    # Replace any invalid characters with underscores
    collection_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', collection_name)

    # Ensure that the name starts and ends with an alphanumeric character
    if not collection_name[0].isalnum():
        collection_name = 'X' + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + 'X'

    return collection_name

def remove_txt_extension(filename):
    if filename.endswith('.txt'):
        return filename[:-4]
    else:
        return filename

def find_name_in_dict(name_list, name_dict, name):
    for n in name_list:
        if n in name_dict and n == name:
            return name_dict[n]
    return None

def save_audio(temp_path):
    temp_file_path = temp_path
    file_path = "./transctibe_audio.wav"
    if os.path.exists(file_path):
        os.remove(file_path)
    shutil.copy(temp_file_path, file_path)

id = []

def get_voices():
    headers = {
        "xi-api-key": os.environ['TOKEN']
    }
    response = requests.get('https://api.elevenlabs.io/v1/voices', headers=headers)
    response = response.json()
    for i in response['voices']:
        id.append(i)
get_voices()

speaker_name = []
speaker_values = []
for i in id:
    speaker_name.append(i['name'])
    speaker_values.append(i['voice_id'])

def find_voice_by_name(voice_list, name):
    for voice in voice_list:
        if voice.get('name') == name:
            return voice
    return None


def text_to_speech(text,voice_id):
    headers = {
        "xi-api-key": os.environ['TOKEN']
    }
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    data = {
        "text": text,
        "voice_settings": {
        "stability": 0,
        "similarity_boost": 0
        }
    }
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    response = requests.post(url, headers=headers, json=data)
    if os.path.exists("prompt_response.mp3"):
        os.remove("prompt_response.mp3")
    with open('prompt_response.mp3', 'wb') as f:
        f.write(response.content)
    return response.content
    


template = """Given the following extracted parts of a long document and a question, create a final answer. 
If you don't know the answer, just say that you don't know. Try to make up an answer relevent to the book {book} and if something is asked outside of your context simply reply 'I don't know
You are the Fictional character {protagonist} from the story and having a conversation with the human
=========
{context}
=========
{chat_history}
=========
HUMAN: {question}

FINAL ANSWER AS THE CHARACTER HIMSELF:"""

prompt = PromptTemplate(
    input_variables=["chat_history","context","protagonist", "book","question"], 
    template=template
)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
chain = load_qa_chain(OpenAI(temperature=0.7), chain_type="stuff", memory=memory, prompt=prompt)


def chatbot(query,voice_id,protagonist,book,docsearch):
    context = ''
    docs = docsearch.similarity_search(query, include_metadata=True,k=3)
    for i in docs:
        context += i.page_content
    result =chain({"input_documents": docs,"protagonist":protagonist, "book" : book,"question":query}, return_only_outputs=True)
   
    response = text_to_speech(result['output_text'],voice_id)
    prompt_response_speech = "prompt_response.mp3"
    return prompt_response_speech
