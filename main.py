from flask import Flask, render_template, request,send_file, send_from_directory, Response
import subprocess
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
# import RPi.GPIO as GPIO
import time

models = ["apple.pt", "banana.pt", "orange.pt"]
fruitc = 0
sensor_data = {"sensor": 0, "humid": 0, "temp": 0}

app = Flask(__name__, static_url_path='/static')
servoPIN = 17
mq2_pin = 23
mq15_pin = 10
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(servoPIN, GPIO.OUT)
# GPIO.setup(mq2_pin, GPIO.IN)
# GPIO.setup(mq15_pin, GPIO.IN)
count = 0


# @app.route('/move')
# def move_camera():
#     angle = request.args.get('angle')
#     angle = int(angle)
#     p = GPIO.PWM(servoPIN, 50)  # GPIO 17 for PWM with 50Hz
#     p.start(2.5)  # Initialization
#     try:
#         while True:
#             p.ChangeDutyCycle(angle / 18.0) + 2.5
#             time.sleep(2)
#     except KeyboardInterrupt:
#         p.stop()
#         GPIO.cleanup()


@app.route("/pred")
def make_prediction():
    _c = cv2.VideoCapture(0)
    # Load the model
    interpreter = tflite.Interpreter(model_path="model_densenet.tflite")
    # interpreter = tflite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()
    _, img = _c.read()
    img = cv2.resize(img, (128, 128), cv2.INTER_AREA)
    input_tensor = np.array(np.expand_dims(img, 0), dtype=np.float32)
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_details)
    pred = np.squeeze(output_data)
    print(pred)
    fruit += 1
    if (fruit >= 3):
        fruit = 0
    return str(pred)


@app.route("/getFruit")
def send_f():
    global fruitc
    return {"curr": fruitc}


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
def send_report():
    return send_file("freshness/out/input.png")


@app.route("/hook", methods=['POST'])
def get_sensor_data():
    global sensor_data
    print("Got Data From ESP8266")
    sensor_data["sensor"] = request.get_json()['sensor']
    sensor_data["humid"] = request.get_json()['humid']
    sensor_data["temp"] = request.get_json()['temp']
    return "thanks"


@app.route("/sensor")
def put_sensor_data():
    mq2_val = GPIO.input(mq2_pin)
    mq15_val = GPIO.input(mq15_pin)
    return render_template('sensor.html',
                           sensor=sensor_data["sensor"],
                           temp=sensor_data["temp"],
                           humid=sensor_data["humid"])


@app.route('/detect', methods=['GET', 'POST'])
def detect_image():
    global fruitc
    fruit = models[fruitc]
    capture()
    subprocess.run(["rm", "static/input.png"])
    subprocess.run(["rm", "-rf", "freshness/out"])
    subprocess.run([
        "python3", "detect.py", "--weights", fruit, "--source", "input.png",
        "--hide-conf", "--project", "freshness", "--name", "out"
    ])
    subprocess.run(["cp", "freshness/out/input.png", "static"])
    fruitc += 1
    if (fruitc >= 3):
        fruitc = 0
    return render_template('out.html')


@app.route("/")
def render_main():
    return render_template('index.html')


app.run(host='0.0.0.0', port=5000)
