from flask import Flask, render_template, request, send_from_directory, Response
import subprocess
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import RPi.GPIO as GPIO
import time

app = Flask(__name__, static_url_path='/static')
servoPIN = 17
mq2_pin = 23
mq15_pin = 10
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)
GPIO.setup(mq2_pin, GPIO.IN)
GPIO.setup(mq15_pin, GPIO.IN)


@app.route('/move')
def move_camera():
    p = GPIO.PWM(servoPIN, 50)  # GPIO 17 for PWM with 50Hz
    p.start(2.5)  # Initialization
    try:
        while True:
            p.ChangeDutyCycle(2.5)
            time.sleep(2)
            p.ChangeDutyCycle(5)
            time.sleep(2)
            p.ChangeDutyCycle(7.5)
            time.sleep(2)
            p.ChangeDutyCycle(10)
            time.sleep(2)
            p.ChangeDutyCycle(12.5)
            time.sleep(2)
            p.ChangeDutyCycle(10)
            time.sleep(2)
            p.ChangeDutyCycle(7.5)
            time.sleep(2)
            p.ChangeDutyCycle(5)
            time.sleep(2)
            p.ChangeDutyCycle(2.5)
            time.sleep(2)
    except KeyboardInterrupt:
        p.stop()
        GPIO.cleanup()


@app.route("/pred")
def make_prediction():
    _c = cv2.VideoCapture(0)
    # Load the model
    interpreter = tflite.Interpreter(model_path="model_densenet.tflite")
    # interpreter = tflite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()
    _, img = _c.read()
    img = cv2.resize(img, (128, 128))
    input_tensor = np.array(np.expand_dims(img, 0), dtype=np.float32)
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = np.squeeze(output_data)
    print(pred)
    return str(pred)


# def gen_frames():
#     while True:
#         succ, frame = cam.read()
#         if not succ:
#             break
#         else:
#             ret, buff = cv2.imencode('.jpg', frame)
#             frame = buff.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) +
#                    b'\r\n')

# @app.route('/live')
# def video_feed():
#     return Response(gen_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


def capture():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    subprocess.run(["rm", "./input.png"])
    if (ret):
        cv2.imwrite('./input.png', frame)
    cam.release()
    return "Photo Done"


@app.route('/img')
def send_report(path):
    return send_from_directory("freshness", path)


@app.route("/sensor")
def get_sensor_data():
    mq2_val = GPIO.input(mq2_pin)
    mq15_val = GPIO.input(mq15_pin)
    return str(mq2_val) + str(mq15_val)


@app.route("/detect", methods=['GET', 'POST'])
def detect_image():
    fruit = request.form['fruit']
    capture()
    subprocess.run(["rm", "static/input.png"])
    subprocess.run(["rm", "-rf", "freshness/out"])
    subprocess.run([
        "python3", "detect.py", "--weights", fruit + ".pt", "--source",
        "input.png", "--hide-conf", "--project", "freshness", "--name", "out"
    ])
    subprocess.run(["cp", "freshness/out/input.png", "static"])
    return render_template('out.html')


@app.route("/")
def render_main():
    return render_template('index.html')


app.run(host='0.0.0.0', port=5000)
