from flask import render_template, request
from app.model import load_model

# Load model and tokenizer at the start
model, tokenizer = load_model()

def configure_routes(app):
    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            features = request.form['features']
            description = generate_description(features)
            return render_template('index.html', description=description)
        return render_template('index.html', description=None)

def generate_description(features):
    input_text = "generate description: " + features
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    output = model.generate(**inputs)
    description = tokenizer.decode(output[0], skip_special_tokens=True)
    return description
