from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests

app = Flask(__name__)

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Helper function to send image to external endpoints
def send_to_external_services(image):
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()

    # Prepare requests to external endpoints
    responses = []

    try:
        response1 = requests.post(
            'https://7721-202-51-197-51.ngrok-free.app/recognize_vehicle',
            files={'image': ('image.jpg', img_bytes, 'image/jpeg')}
        )
        responses.append(response1.json() if response1.status_code == 200 else {'error': 'vehicle not recognized'})
    except Exception as e:
        responses.append({'error': str(e)})

    try:
        response2 = requests.post(
            'https://e631-2001-448a-2061-2ed8-512b-9b7a-f245-3bb3.ngrok-free.app/detect',
            files={'image': ('image.jpg', img_bytes, 'image/jpeg')}
        )
        responses.append(response2.json() if response2.status_code == 200 else {'error': 'fruit not recognized'})
    except Exception as e:
        responses.append({'error': str(e)})

    try:
        response3 = requests.post(
            'https://aaaf-180-243-3-96.ngrok-free.app/electronics',
            files={'image': ('image.jpg', img_bytes, 'image/jpeg')}
        )
        responses.append(response3.json() if response3.status_code == 200 else {'error': 'electronics not recognized'})
    except Exception as e:
        responses.append({'error': str(e)})

    return responses

# Route for uploading image via HTML form
@app.route('/')
def upload_form():
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Age and Gender Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #6e8efb, #a777e3);
            color: #fff;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
            overflow: hidden;
        }
        header {
            text-align: center;
            margin-bottom: 30px;
        }
        header h1 {
            font-size: 2.5rem;
            margin: 0;
        }
        header p {
            font-size: 1.2rem;
            margin-top: 10px;
        }
        .content {
            display: flex;
            align-items: flex-start;
            gap: 20px;
        }
        .form-container {
            background: rgba(255, 255, 255, 0.2);
            padding: 20px 30px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            text-align: center;
        }
        .form-container input[type="file"] {
            display: block;
            margin: 15px auto;
            padding: 10px;
            border: none;
            border-radius: 5px;
            background: #fff;
            color: #333;
        }
        .form-container button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 1rem;
            cursor: pointer;
            transition: background 0.3s ease;
        }
        .form-container button:hover {
            background: #45a049;
        }
        /* Style for result box */
        .results-container {
            margin-top: 20px;
            background: rgba(255, 255, 255, 0.2);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            color: #000;
            max-width: 400px;
            word-wrap: break-word;
            max-height: 300px;  /* Height limit */
            overflow-y: auto;  /* Scrollbar if content exceeds limit */
        }

        /* Styling scrollbar */
        .results-container::-webkit-scrollbar {
            width: 8px;
        }

        .results-container::-webkit-scrollbar-thumb {
            background-color: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
        }

        .results-container::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
        }

        footer {
            margin-top: 20px;
            text-align: center;
            font-size: 0.9rem;
        }
        .gif-container img {
            max-width: 200px;
            border-radius: 10px;
        }

        /* Style for cursor trail */
        .trail {
            position: absolute;
            width: 12px;
            height: 12px;
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 50%;
            pointer-events: none;
            animation: trailAnimation 1s forwards;
            z-index: 1000;
        }

        @keyframes trailAnimation {
            0% {
                transform: scale(1);
                opacity: 1;
            }
            100% {
                transform: scale(0);
                opacity: 0;
            }
        }
    </style>
</head>
<body>
    <header>
        <h1>Age and Gender Prediction</h1>
        <p>Upload an image to predict age and gender using AI-powered analysis</p>
    </header>
    <div class="content">
        <div class="form-container">
            <form id="predict-form" enctype="multipart/form-data" onsubmit="showResults(event);">
                <input type="file" name="image" accept="image/*" required>
                <button type="submit">Predict</button>
            </form>
            <div class="results-container" id="results">
                <strong>Results:</strong>
                <pre id="resultsBox">No results yet.</pre>
            </div>
        </div>
        <div class="gif-container">
            <img src="https://media.tenor.com/Rp0U7bdOhSUAAAAj/anime.gif" alt="Anime GIF">
        </div>
    </div>
    <footer>
        <p>&copy; 2024 AI Prediction Service. All rights reserved.</p>
    </footer>

    <div class="cursor-trail" id="cursorTrail"></div>

    <script>
         // Function for handling cursor trail
        const trails = [];

        function createTrail(x, y) {
            const trail = document.createElement('div');
            trail.classList.add('trail');
            document.body.appendChild(trail);
            trail.style.left = `${x - 6}px`; // Adjust to center the trail
            trail.style.top = `${y - 6}px`; // Adjust to center the trail

            // Remove the trail after animation ends
            setTimeout(() => {
                trail.remove();
            }, 1000);
        }

        // Update trail on mouse move
        document.addEventListener('mousemove', (e) => {
            createTrail(e.pageX, e.pageY);
        });

        // Show the results from the backend
        async function showResults(event) {
            event.preventDefault(); // Prevent form from submitting

            const form = document.getElementById("predict-form");
            const formData = new FormData(form);

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });

                const resultBox = document.getElementById("resultsBox");

                if (response.ok) {
                    const data = await response.json();
                    resultBox.textContent = JSON.stringify(data, null, 2);
                } else {
                    resultBox.textContent = "Error processing the image.";
                }
            } catch (error) {
                const resultBox = document.getElementById("resultsBox");
                resultBox.textContent = `Error: ${error.message}`;
            }
        }
    </script>
</body>
</html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Read the image from the request
    file = request.files['image']
    image = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Invalid image'}), 400

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Send image to external services regardless of face detection
    external_results = send_to_external_services(frame)

    # Process detected faces locally if any
    results = []
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]

            # Preprocess face for predictions
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            # Predict gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            # Predict age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]

            # Add result for this face
            results.append({
                'gender': gender,
                'age': age
            })

    return jsonify({
        'results': results,
        'external_service_results': external_results
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
