# Import necessary libraries
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
from model import CNNModel  
from utils import test_val_transform  
import threading
from training import train_model
from utils import train_loader, val_loader, test_loader
from training import criteria, optimizer, num_epochs, scheduler, device, validate_model, evaluate_on_test_set
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Initialize Flask application
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your trained model
model = CNNModel().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Route for the home page
@app.route('/')
def home():
    return render_template('home.html')  

# Route for uploading images
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            return redirect(url_for('predict', filename=filename))
    return render_template('upload.html')

# Route for making predictions
@app.route('/predict/<filename>')
def predict(filename):
    
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image = Image.open(image_path)
    image = test_val_transform(image) 
    image = image.unsqueeze(0)  
    image = image.to(device)

    with torch.no_grad():
        prediction = model(image)
        predicted_label = prediction.item() > 0.5 

    result = "The image is predicted to have pneumonia." if predicted_label else "The image is predicted to not have pneumonia."
    return render_template('result.html', result=result)

# Route for training the model
@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        # Start the training process
        train_thread = threading.Thread(target=train_model, args=(model, train_loader, val_loader, criteria, optimizer, scheduler, device, num_epochs))
        train_thread.start()
        return render_template('training_started.html')  
    return render_template('train.html') 


# Route for evaluating on the validation set
@app.route('/evaluate_validation')
def evaluate_validation():
    avg_val_loss, accuracy = validate_model(model, val_loader, criteria, device)
    return render_template('evaluation_valresult.html', result=f'Average Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')

# Route for evaluating on the test set
@app.route('/evaluate_test')
def evaluate_test():
    metrics = evaluate_on_test_set(model, test_loader, criteria, device)
    cache_buster = datetime.now().strftime("%Y%m%d%H%M%S")
    return render_template('evaluation_result.html', metrics=metrics, cache_buster=cache_buster)

@app.route('/training_results')
def training_results():
    try:
        with open('./training_results.txt', 'r') as file:
            results = file.read()
    except FileNotFoundError:
        results = "Training results not found."
    return render_template('training_results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)