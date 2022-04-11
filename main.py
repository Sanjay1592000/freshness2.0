from flask import Flask, render_template, request
import subprocess
import cv2

app = Flask(__name__)

@app.route("/takepic")
def capture():
    cap = cv2.VideoCapture(0)
    _,frame = cap.read() 
    cv2.imwrite('./input.png',frame)
    cap.release()
    return "Photo Done"


@app.route("/detect",methods=['GET','POST'])
def detect_image():
    fruit = request.form['fruit']
    subprocess.run(["python3",
                   "detect.py",
                   "--weights",
                   fruit+".pt",
                   "--source",
                   "input.png",
                   "--project",
                   "freshness",
                   "--name",
                   "out"])
    return "Detectiong done"

@app.route("/show")
def show_image():
    return render_template('out.html')


@app.route("/")
def render_main():
    return render_template('index.html')

app.run(host='0.0.0.0', port=5000)