from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from fairy_tale_generation import gen_run, generate_story, load_lists
import threading

app = Flask(__name__, static_folder='../static', template_folder='../templates')
socketio = SocketIO(app, cors_allowed_origins='*', async_mode="eventlet")

@app.route('/')
def fairytale():
    return render_template('fairytale.html')

@socketio.on('connect')
def handle_connect():
    thread = threading.Thread(target=gen_run)
    thread.start()
    load_lists(socketio)
    socketio.emit("gen_run_done","동화 생성 ai 준비됨.")

@socketio.on('start_story')
def handle_start_story(data):
    keywords = data.get('keywords')
    readage = data.get('readage')
    if not keywords or not readage:
        socketio.emit('story_error', {'message': '키워드나 연령이 입력되지 않았습니다.'})
        return
    try:
        generate_story(keywords, readage, socketio)
        load_lists(socketio)
    except Exception as e:
        socketio.emit("story_error", f"동화 생성 중 오류가 발생했습니다")

"""@socketio.on('load_fairytale')
def handle_load_fairytale(data):
    title = data.get('title')
    load_fairytale(title, socketio)"""

if __name__ == "__main__":
    socketio.run(app, debug=True, host='127.0.0.1', port=5000)