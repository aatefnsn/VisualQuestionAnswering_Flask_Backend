from flask import Flask, request, jsonify
from six.moves import cPickle as pickle
#from app.torch_utils import transform_image, get_prediction, transform_question_BERT#, transform_question, transform_question_two
from torch_utils import transform_image, get_prediction, transform_question_BERT#, transform_question, transform_question_two
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        qu = request.form.get('question')
        if (qu == ""):
            return jsonify({'error': 'no question'})

        try:
            print('inside try block')
            img = Image.open(file)
            img = img.convert('RGB')
            print('Calculating Qtensor')
            Qtensor_mod = transform_question_BERT(qu)
            print('Calculating Itensor')
            Itensor = transform_image(img)
            prediction = get_prediction(Itensor,Qtensor_mod)

            objects = []
            with (open("app/i2a.pkl", "rb")) as openfile:
                while True:
                    try:
                        objects.append(pickle.load(openfile))
                    except EOFError:
                        break

            """data = {'prediction-0': prediction[0][0].item(), 'class_name-0': str(class_name0),
                    'prediction-1': prediction[0][1].item(), 'class_name-1': str(class_name1),
                    'prediction-2': prediction[0][2].item(), 'class_name-2': str(class_name2),
                    'prediction-3': prediction[0][3].item(), 'class_name-3': str(class_name3),
                    'prediction-4': prediction[0][4].item(), 'class_name-4': str(class_name4),

                    'prediction-5': prediction[0][5].item(), 'class_name-5': str(class_name5),
                    'prediction-6': prediction[0][6].item(), 'class_name-6': str(class_name6),
                    'prediction-7': prediction[0][7].item(), 'class_name-7': str(class_name7),
                    'prediction-8': prediction[0][8].item(), 'class_name-8': str(class_name8),
                    'prediction-9': prediction[0][9].item(), 'class_name-9': str(class_name9),

                    'prediction-10': prediction[0][10].item(), 'class_name-10': str(objects[0][prediction[0][10].item()]),
                    'prediction-11': prediction[0][11].item(), 'class_name-11': str(objects[0][prediction[0][11].item()]),
                    'prediction-12': prediction[0][12].item(), 'class_name-12': str(objects[0][prediction[0][12].item()]),
                    'prediction-13': prediction[0][13].item(), 'class_name-13': str(objects[0][prediction[0][13].item()]),
                    'prediction-14': prediction[0][14].item(), 'class_name-14': str(objects[0][prediction[0][14].item()]),


                    'prediction-999': prediction[0][999].item(), 'class_name-999': str(class_name999),
                    'prediction-998': prediction[0][998].item(), 'class_name-998': str(class_name998),
                    'prediction-997': prediction[0][997].item(), 'class_name-997': str(class_name997),
                    'prediction-996': prediction[0][996].item(), 'class_name-996': str(class_name996),
                    'prediction-995': prediction[0][995].item(), 'class_name-995': str(class_name995)
            }
            """
            data = {'class_name-0': str(objects[0][prediction[0][0].item()]),
                    'class_name-1': str(objects[0][prediction[0][1].item()]),
                    'class_name-2': str(objects[0][prediction[0][2].item()]),
                    'class_name-3': str(objects[0][prediction[0][3].item()]),
                    'class_name-4': str(objects[0][prediction[0][4].item()]),
                    'class_name-5': str(objects[0][prediction[0][5].item()]),
                    'class_name-6': str(objects[0][prediction[0][6].item()]),
                    'class_name-7': str(objects[0][prediction[0][7].item()]),
                    'class_name-8': str(objects[0][prediction[0][8].item()]),
                    'class_name-9': str(objects[0][prediction[0][9].item()]),
                    'class_name-998': str(objects[0][prediction[0][998].item()]),
                    'class_name-999': str(objects[0][prediction[0][999].item()])
            }
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})

if __name__ == "__main__":
    #port = os.environ.get("PORT", 5000)
    app.run(debug=False)#, host="0.0.0.0", port=port)
