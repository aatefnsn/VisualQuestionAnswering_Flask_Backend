import requests
import json
import os
from six.moves import cPickle as pickle
newHeaders = {'Content-type': 'multipart/form-data'}#, 'Accept': 'text/plain'}
# https://your-heroku-app-name.herokuapp.com/predict
# http://localhost:5000/predict
#resp = requests.post("http://localhost:5000/predict", files={'file': open('eight.png', 'rb')})
my_question = 'what color is the dog?'
myobj = {'somekey': 'somevalue'}
#file = '/Users/ahmed_nada/pytorch-flask/app/COCO_train2014_000000002056.jpg'
payload = {"question": "what is he riding?"}
#payload = "what is he riding?"
#files = {'json': (None, payload, 'application/json'), 'file': (os.path.basename(file), open(file, 'rb'), 'application/octet-stream')}
#myobj1 = json.dumps(myobj)
#json_data = json.dumps(my_question)
#data = {'question': 1234}
#resp = requests.post("http://localhost:5000/predict", files={'file': open('COCO_train2014_000000000394.jpg','rb')})
#resp = requests.post("http://localhost:5000/predict", files={'file': open('COCO_train2014_000000000081.jpg','rb')})
#resp = requests.post("http://localhost:5000/predict", files={'file': open('COCO_train2014_000000000532.jpg','rb')})
#resp = requests.post("https://predict-mpfalbvdhq-uw.a.run.app/predict", files={'file': open('/Users/ahmed_nada/test/COCO_train2014_000000000532.jpg','rb')}, data = {'question': 'what color is the bus?'})
#resp = requests.post("https://predict-mpfalbvdhq-uw.a.run.app/predict", files={'file': open('/Users/ahmed_nada/test/COCO_train2014_000000002150.jpg','rb')}, data = {'question': 'what is holding the flowers?'})
resp = requests.post("https://predict-bc7ximz6ca-zf.a.run.app/predict", files={'file': open('/Users/ahmed_nada/test/COCO_train2014_000000002150.jpg','rb')}, data = {'question': 'what is holding the flowers?'})
#resp = requests.post("http://localhost:5000/predict", files={'file': open('/Users/ahmed_nada/test/COCO_train2014_000000000532.jpg','rb')}, data = {'question': 'what color is the bus?'})
#resp = requests.post("http://localhost:5000/predict", files={'json': (None, json.dumps("what animal is he riding?"), 'application/json') ,'file': open('COCO_train2014_000000002056.jpg','rb')})
#, data = {u'question': u"ya rab?"})#, headers=newHeaders)#, json=json.dumps({'question': "ya rab?"}))#, headers=newHeaders)#{'question':'what is the animal?'})
#resp = requests.post("http://localhost:5000/predict", files={'file': open('/Users/ahmed_nada/pytorch-flask/app/cam.jpeg','rb')})
#resp = requests.post("http://localhost:5000/predict", files={'file': open('COCO_train2014_000000002150.jpg','rb')})
#resp = requests.post("http://localhost:5000/predict", files={'file': open('soc.jpeg','rb')})
#resp = requests.post("http://localhost:5000/predict", files={'file': open('COCO_val2014_000000000428.jpg','rb')})
#resp = requests.post("http://localhost:5000/predict", files={'file': open('COCO_train2014_000000002178.jpg','rb')})
#resp = requests.post("http://localhost:5000/predict", files={'file': open('dog.png','rb')})
#resp = requests.post("http://localhost:5000/predict", files={'file': open('umbrella.jpeg','rb')})
#resp = requests.post("http://localhost:5000/predict", files={'file': open('ten.jpeg','rb')})
#resp = requests.post("http://localhost:5000/predict", files={'file': open('elephant.jpeg','rb')})
#resp = requests.post("http://localhost:5000/predict", files={'file': open('eat2.jpeg','rb')})
#resp = requests.post("http://localhost:5000/predict", files={'file': open('Zebra.jpeg','rb')})
#resp = requests.post("http://localhost:5000/predict", files={'file': open('cat2.jpeg','rb')})
#resp = requests.post("http://localhost:5000/predict", files={'file': open('dog.png','rb')}, form={'question': 'what animal is this?'})
#resp = requests.post("http://localhost:5000/predict", files={'file': open('dog.png','rb')}, data=json.dumps(list(data)))
#resp = requests.post("http://localhost:5000/predict", files={'file': open('dog.png','rb')},data={'question':'what is the animal name?'})
#data=json.dumps(my_data)
"""
objects = []
with (open("i2a.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
    print(objects)
"""
print(resp.text)
