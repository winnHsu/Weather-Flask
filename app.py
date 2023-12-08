from flask import Flask, jsonify
from flask_cors import CORS
from LSTM import json_result

app = Flask(__name__)
CORS(app)

@app.route('/data', methods=['GET'])
def get_data():
    try:
        print("Data endpoint hit")
        return jsonify(json_result)
    except Exception as e:
        print("Error: ", e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
