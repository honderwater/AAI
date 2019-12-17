from geventwebsocket.handler import WebSocketHandler
from gevent.pywsgi import WSGIServer
from flask import Flask, request, render_template
from time import sleep
import base64
import json
import numpy as np
import cv2

#from tensorflow.keras.models import load_model
#model_ConvNet = load_model('modelo') #substituir pelo ficheiro que guardaste o teu modelo
app = Flask(__name__, template_folder = 'static') #não é preciso mudar, é apenas o sitio onde vais meter o html, se não ele vai como default à pasta template

@app.route('/')
def index():
    return render_template('caligrafia.html') #ficheiro html onde o servidor vai buscar info 

@app.route('/camera')
def camera():

    if request.environ.get('wsgi.websocket'):
        ws = request.environ['wsgi.websocket']
        success = False
        
        while True:
            cam = cv2.VideoCapture(0)#mudar aqui o numero da camara(0 deve ser a camara do pc), tenta 1
            while True:
                image = cam.read()[1]
                buffer = cv2.imencode('.jpg', image)[1].tobytes()
                live_as_text = base64.b64encode(buffer).decode()
                ws.send('{"number":"' + str(live_as_text) + '"}')
                sleep(0.01)
                resp = ws.receive()
                py_resp = json.loads(resp)
                if(py_resp["aqc"]):
                    cam.release()
                    break

            #process image
            success, number_pred = Image_Pre_Processing(image) #Image_Pre_Processing() subsittuir pela vosssa funcao que te 
            #envia o numero que a rede calculou...neste caso retorna um boleano (tipo conseguiu identificar) e a predict do numero que estao a ver
            
            if(success):        
                if(py_resp["waiting"]):
                    ws.send('{"prediction":"' + str(number_pred))
                else:
                    ws.send('{"prediction":"erro"}')
            else:
                resp = ws.receive()
                py_resp = json.loads(resp)
                
                ws.send('{"prediction":"erro"}')

if __name__ == '__main__':
    http_server = WSGIServer(('',9999), app, handler_class=WebSocketHandler)
    http_server.serve_forever()