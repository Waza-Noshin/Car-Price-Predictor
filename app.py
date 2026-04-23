from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
print("Loading model...")
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("Model loaded!")

def predict_price(features):
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    prediction = model.predict(features_scaled)[0]
    return round(prediction, 2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        features = [
            float(data['hp']),
            float(data['mpg']),
            float(data['volume']),
            float(data['weight']),
            float(data['cylinders'])
        ]
        
        price = predict_price(features)
        
        return jsonify({
            'success': True,
            'price': price,
            'formatted_price': f"${price:,.2f}"
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    import webbrowser
    webbrowser.open('http://127.0.0.1:5000')
    app.run(debug=True, port=5000)