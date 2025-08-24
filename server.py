from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app)

# Store processed files with timestamps
processed_files = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_status', methods=['POST'])
def update_status():
    data = request.get_json()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    file_entry = {"file": data["file"], "status": data["status"], "timestamp": timestamp}

    # Update if file exists, else add new entry
    for entry in processed_files:
        if entry["file"] == data["file"]:
            entry["status"] = data["status"]
            entry["timestamp"] = timestamp
            break
    else:
        processed_files.append(file_entry)

    # Send update to clients
    socketio.emit('status_update', file_entry)
    return {"message": "Status updated"}, 200

@app.route('/get_status')
def get_status():
    """Return all processed files and statuses as JSON."""
    return jsonify(processed_files)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8888, debug=True)
