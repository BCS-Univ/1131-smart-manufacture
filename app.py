from flask import Flask, jsonify
from training import train_model

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def trigger_training():
    try:
        loss = train_model()
        return jsonify({
            'status': 'success',
            'message': 'Training completed',
            'final_loss': loss
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
