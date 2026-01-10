from flask import Flask, request, jsonify
from six.moves import cPickle as pickle
#from app.torch_utils import transform_image, get_prediction, transform_question_BERT#, transform_question, transform_question_two
from torch_utils import transform_image, get_prediction, transform_question_BERT#, transform_question, transform_question_two
from PIL import Image
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import json
from datetime import datetime
from azure.eventhub import EventHubProducerClient, EventData

app = Flask(__name__)
CORS(app)

# Event Hub configuration
EVENT_HUB_CONNECTION_STRING = os.getenv('EVENT_HUB_CONNECTION_STRING')
EVENT_HUB_NAME = os.getenv('EVENT_HUB_NAME', 'vqa-predictions')

# Question type keywords
QUESTION_TYPE_KEYWORDS = {
    'color': ['color', 'what color', 'colored', 'colour'],
    'object': ['what', 'what is', 'what are', 'object', 'objects'],
    'count': ['how many', 'count', 'number of'],
    'location': ['where', 'left', 'right', 'behind', 'front', 'position', 'located'],
    'action': ['is', 'are', 'doing', 'wearing', 'holding', 'action'],
    'yes_no': ['is there', 'are there', 'do', 'does', 'can', 'will']
}

def categorize_question(question):
    """Categorize question by type based on keywords"""
    q_lower = question.lower()
    
    # Check each category
    for category, keywords in QUESTION_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in q_lower:
                return category
    
    return 'other'

def log_prediction_to_event_hub(question, question_type, top_answer, top_probability, model_version='v1'):
    """Log prediction to Azure Event Hub for real-time dashboard"""
    try:
        if not EVENT_HUB_CONNECTION_STRING:
            print("⚠️ Event Hub connection string not configured, skipping logging")
            return
        
        producer = EventHubProducerClient.from_connection_string(
            conn_str=EVENT_HUB_CONNECTION_STRING,
            eventhub_name=EVENT_HUB_NAME
        )
        
        event_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'question': question,
            'question_type': question_type,
            'top_answer': str(top_answer),
            'top_probability': float(top_probability),
            'model_version': model_version,
            'user_session_id': request.remote_addr
        }
        
        producer.send_batch([EventData(json.dumps(event_data))])
        producer.close()
        print(f"✓ Logged to Event Hub: {question_type} → {top_answer} ({top_probability*100:.1f}%)")
        
    except Exception as e:
        print(f"⚠️ Event Hub logging failed (non-critical): {e}")
        # Don't fail prediction if logging fails

# Initialize model on startup with detailed logging
print("=" * 60)
print("VQA Backend Initialization")
print("=" * 60)
try:
    from torch_utils import get_model
    print("Loading VQA model...")
    model = get_model()
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ FATAL: Failed to load model during startup: {e}")
    import traceback
    traceback.print_exc()
print("=" * 60)

# Rate limiting: 100 requests per day per IP address
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day"],
    storage_uri="memory://"
)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for deployment monitoring"""
    return jsonify({
        'status': 'healthy',
        'service': 'vqa-backend'
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Enhanced predict endpoint that returns predictions with probabilities
    for Databricks logging and monitoring.
    
    Request:
        file: Image file (png, jpg, jpeg)
        question: Question string
        
    Response:
        {
            'predicted_answers': [
                {
                    'rank': 1,
                    'class_id': int,
                    'class_name': str,
                    'probability': float,
                    'confidence': str
                },
                ...
            ]
        }
    """
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
            print('Inside predict endpoint - processing request')
            
            # Load image and convert to RGB
            img = Image.open(file)
            img = img.convert('RGB')
            print('✓ Image loaded and converted to RGB')
            
            # Transform question using BERT tokenizer
            print('Transforming question with BERT...')
            Qtensor_mod = transform_question_BERT(qu)
            print('✓ Question tensor created')
            
            # Transform image using ResNet18 feature extractor
            print('Transforming image with ResNet18...')
            Itensor = transform_image(img)
            print('✓ Image tensor created')
            
            # Get predictions with probabilities (sorted by confidence)
            print('Getting model predictions...')
            predictions = get_prediction(Itensor, Qtensor_mod)
            print('✓ Predictions received from model')
            
            # Load answer index mapping (i2a.pkl)
            print('Loading answer index mapping...')
            objects = []
            with (open("app/i2a.pkl", "rb")) as openfile:
                while True:
                    try:
                        objects.append(pickle.load(openfile))
                    except EOFError:
                        break
            print('✓ Answer mapping loaded')
            
            # Build response with predictions and probabilities
            predicted_answers = []
            for rank, pred in enumerate(predictions[:20]):  # Top 20 predictions
                class_id = pred['class_id']
                class_name = objects[0][class_id] if class_id < len(objects[0]) else f"unknown-{class_id}"
                
                predicted_answers.append({
                    'rank': rank + 1,
                    'class_id': class_id,
                    'class_name': str(class_name),
                    'probability': pred['probability'],
                    'confidence': pred['confidence']
                })
            
            # Get top prediction
            top_pred = predicted_answers[0]
            top_answer = top_pred['class_name']
            top_probability = top_pred['probability']
            
            # Categorize question and log to Event Hub
            question_type = categorize_question(qu)
            log_prediction_to_event_hub(
                question=qu,
                question_type=question_type,
                top_answer=top_answer,
                top_probability=top_probability,
                model_version='v1'
            )
            
            print(f'✓ Returning {len(predicted_answers)} predictions')
            response = {
                'status': 'success',
                'predicted_answers': predicted_answers
            }
            return jsonify(response), 200
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f'✗ ERROR during prediction: {str(e)}')
            print(error_trace)
            return jsonify({
                'error': 'error during prediction',
                'details': str(e),
                'trace': error_trace
            }), 500


# =====================================================================
# OLD PREDICT FUNCTION (COMMENTED OUT FOR REFERENCE)
# =====================================================================
# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         file = request.files.get('file')
#         if file is None or file.filename == "":
#             return jsonify({'error': 'no file'})
#         if not allowed_file(file.filename):
#             return jsonify({'error': 'format not supported'})
#
#         qu = request.form.get('question')
#         if (qu == ""):
#             return jsonify({'error': 'no question'})
#
#         try:
#             print('inside try block')
#             img = Image.open(file)
#             img = img.convert('RGB')
#             print('Calculating Qtensor')
#             Qtensor_mod = transform_question_BERT(qu)
#             print('Calculating Itensor')
#             Itensor = transform_image(img)
#             prediction = get_prediction(Itensor,Qtensor_mod)
#
#             objects = []
#             with (open("app/i2a.pkl", "rb")) as openfile:
#                 while True:
#                     try:
#                         objects.append(pickle.load(openfile))
#                     except EOFError:
#                         break
#
#             """data = {'prediction-0': prediction[0][0].item(), 'class_name-0': str(class_name0),
#                     'prediction-1': prediction[0][1].item(), 'class_name-1': str(class_name1),
#                     'prediction-2': prediction[0][2].item(), 'class_name-2': str(class_name2),
#                     'prediction-3': prediction[0][3].item(), 'class_name-3': str(class_name3),
#                     'prediction-4': prediction[0][4].item(), 'class_name-4': str(class_name4),
#
#                     'prediction-5': prediction[0][5].item(), 'class_name-5': str(class_name5),
#                     'prediction-6': prediction[0][6].item(), 'class_name-6': str(class_name6),
#                     'prediction-7': prediction[0][7].item(), 'class_name-7': str(class_name7),
#                     'prediction-8': prediction[0][8].item(), 'class_name-8': str(class_name8),
#                     'prediction-9': prediction[0][9].item(), 'class_name-9': str(class_name9),
#
#                     'prediction-10': prediction[0][10].item(), 'class_name-10': str(objects[0][prediction[0][10].item()]),
#                     'prediction-11': prediction[0][11].item(), 'class_name-11': str(objects[0][prediction[0][11].item()]),
#                     'prediction-12': prediction[0][12].item(), 'class_name-12': str(objects[0][prediction[0][12].item()]),
#                     'prediction-13': prediction[0][13].item(), 'class_name-13': str(objects[0][prediction[0][13].item()]),
#                     'prediction-14': prediction[0][14].item(), 'class_name-14': str(objects[0][prediction[0][14].item()]),
#
#
#                     'prediction-999': prediction[0][999].item(), 'class_name-999': str(class_name999),
#                     'prediction-998': prediction[0][998].item(), 'class_name-998': str(class_name998),
#                     'prediction-997': prediction[0][997].item(), 'class_name-997': str(class_name997),
#                     'prediction-996': prediction[0][996].item(), 'class_name-996': str(class_name996),
#                     'prediction-995': prediction[0][995].item(), 'class_name-995': str(class_name995)
#             }
#             """
#             data = {'class_name-0': str(objects[0][prediction[0][0].item()]),
#                     'class_name-1': str(objects[0][prediction[0][1].item()]),
#                     'class_name-2': str(objects[0][prediction[0][2].item()]),
#                     'class_name-3': str(objects[0][prediction[0][3].item()]),
#                     'class_name-4': str(objects[0][prediction[0][4].item()]),
#                     'class_name-5': str(objects[0][prediction[0][5].item()]),
#                     'class_name-6': str(objects[0][prediction[0][6].item()]),
#                     'class_name-7': str(objects[0][prediction[0][7].item()]),
#                     'class_name-8': str(objects[0][prediction[0][8].item()]),
#                     'class_name-9': str(objects[0][prediction[0][9].item()]),
#                     'class_name-998': str(objects[0][prediction[0][998].item()]),
#                     'class_name-999': str(objects[0][prediction[0][999].item()])
#             }
#             return jsonify(data)
#         except Exception as e:
#             import traceback
#             error_trace = traceback.format_exc()
#             print(f'ERROR during prediction: {str(e)}')
#             print(error_trace)
#             return jsonify({'error': 'error during prediction', 'details': str(e), 'trace': error_trace}), 500
# =====================================================================

if __name__ == "__main__":
    #port = os.environ.get("PORT", 5000)
    app.run(debug=False)#, host="0.0.0.0", port=port)
