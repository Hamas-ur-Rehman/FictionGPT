from flask import Flask, redirect,request
from flask_cors import CORS
from test import *

app = Flask(__name__)
CORS(app)

@app.route('/')
def redirect_to_new_url():
    url = request.host_url
    redirect_url = url.replace(request.host.split(':')[1], '7860')
    gradio_app_interface()
    return redirect(redirect_url)

if __name__ == '__main__':
    app.run(debug=True)
