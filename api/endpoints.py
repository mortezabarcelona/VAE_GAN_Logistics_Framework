from flask import Flask, jsonify
import time
from simulation.live_api_simulation import simulate_vae_gan_dynamic

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    """
    API endpoint that returns a simulated prediction from the VAE-GAN model.
    """
    predicted_cost = simulate_vae_gan_dynamic()
    response = {
        "predicted_cost": predicted_cost,
        "timestamp": time.time()
    }
    return jsonify(response)

if __name__ == '__main__':
    # Run the Flask API on port 5000 in debug mode
    app.run(debug=True, port=5000)
