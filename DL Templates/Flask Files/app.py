from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
import os

app = Flask(__name__)

# Load the model once at the beginning
model = load_model('project1.h5')

@app.route('/predict', methods=["GET", "POST"])
def res():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static/uploads')

        # Ensure the uploads directory exists
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)

        filepath = os.path.join(upload_path, f.filename)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        pred = np.argmax(preds, axis=1)

        index=['air hockey', 'ampute football', 'archery', 'arm wrestling', 'axe throwing',
    'balance beam', 'barell racing', 'baseball', 'basketball', 'baton twirling',
    'bike polo', 'billiards', 'bmx', 'bobsled', 'bowling', 'boxing', 'bull riding',
    'bungee jumping', 'canoe slamon', 'cheerleading', 'chuckwagon racing', 'cricket',
    'croquet', 'curling', 'disc golf', 'fencing', 'field hockey', 'figure skating men',
    'figure skating pairs', 'figure skating women', 'fly fishing', 'football',
    'formula 1 racing', 'frisbee', 'gaga', 'giant slalom', 'golf', 'hammer throw',
    'hang gliding', 'harness racing', 'high jump', 'hockey', 'horse jumping',
    'horse racing', 'horseshoe pitching', 'hurdles', 'hydroplane racing', 'ice climbing',
    'ice yachting', 'jai alai', 'javelin', 'jousting', 'judo', 'lacrosse', 'log rolling',
    'luge', 'motorcycle racing', 'mushing', 'nascar racing', 'olympic wrestling',
    'parallel bar', 'pole climbing', 'pole dancing', 'pole vault', 'polo', 'pommel horse',
    'rings', 'rock climbing', 'roller derby', 'rollerblade racing', 'rowing', 'rugby',
    'sailboat racing', 'shot put', 'shuffleboard', 'sidecar racing', 'ski jumping',
    'sky surfing', 'skydiving', 'snow boarding', 'snowmobile racing', 'speed skating',
    'steer wrestling', 'sumo wrestling', 'surfing', 'swimming', 'table tennis', 'tennis',
    'track bicycle', 'trapeze', 'tug of war', 'ultimate', 'uneven bars', 'volleyball',
    'water cycling', 'water polo', 'weightlifting', 'wheelchair basketball',
    'wheelchair racing', 'wingsuit flying']
        result = index[pred[0]]

        return render_template('output.html', prediction=result)

    return render_template('input.html')

@app.route('/')
def index():
    return render_template('Sporthome.html')

@app.route('/home')
def home():
    return render_template('Sporthome.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

if __name__ == "__main__":
    app.run(debug=False)
