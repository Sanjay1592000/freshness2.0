from flask import Flask, render_template, request, send_from_directory
import subprocess
import cv2

app = Flask(__name__,static_url_path='/static')

def capture():
    cap = cv2.VideoCapture(0)
    _,frame = cap.read() 
    cv2.imwrite('./input.png',frame)
    cap.release()
    return "Photo Done"

@app.route('/img')
def send_report(path):
    return send_from_directory("freshness",path)

@app.route("/detect",methods=['GET','POST'])
def detect_image():
    fruit = request.form['fruit']
    capture()
    subprocess.run(["rm","-rf","freshness/out"])
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
    subprocess.run(["cp","freshness/out/input.png","static"])
    return render_template('out.html')

@app.route("/")
def render_main():
    return render_template('index.html')

app.run(host='0.0.0.0', port=5000)