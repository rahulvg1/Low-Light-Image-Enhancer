from flask import Flask, render_template,jsonify,request
import tensorflow as tf
import keras
from keras.models import load_model
import numpy as np
import base64
from PIL import Image
import io
import cv2

app = Flask(__name__)

def L1(yhat, y):
    loss = np.sum(np.abs(yhat-y),axis = 0)
    return loss

def model():
	global model
	model = load_model('Model/3.1.h5',custom_objects={'L1':L1})
	global graph
	graph = tf.get_default_graph()

model()

def pre(im):
	image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
	img = cv2.resize(image,(256,256))
	#img = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 3, 7) 
	t= []
	image =img/255.
	t.append(image)
	image = np.asarray(t)
	with graph.as_default():
		pred = model.predict(image)
	ip= np.squeeze(pred)*255.
	cv2.imwrite("static/output/out.png",ip)
	return ip


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')


@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    prediction = pre(image)
    response={
    "predicted":True
    }
    return jsonify(response)

    
 
if __name__=='__main__':
	app.run(debug=True)

	#app.run(host='192.168.101.222',debug=True)	

	#app.run(host='0.0.0.0',debug=False)