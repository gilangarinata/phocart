import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

import cv2
import numpy as np
from models import ResnetGenerator
import argparse
from utils import Preprocess
import onnxruntime

UPLOAD_FOLDER = './imageuploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'kuning77'

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory("./images", filename)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/process', methods=['POST','GET'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return {
                "code" : 1210,
                "message" : "No file part" ,
                "path" : None
            }
        file = request.files['file']
        if file.filename == '':
            return {
                "code" : 1211,
                "message" : "no selected file",
                "path" : None 
            }
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = cv2.cvtColor(cv2.imread("./imageuploads/" + filename), cv2.COLOR_BGR2RGB)
            c2p = Photo2CartoonOnxx()
            cartoon = c2p.inference(img)
            if cartoon is not None:
                cv2.imwrite("./images/" + filename, cartoon)
                return {
                    "code" : 2000,
                    "message" : "success",
                    "path" : request.base_url.replace("process","") + "uploads/" + filename
                }
            return {
                "code" : 1212,
                "message" : "No face detected.",
                "path" : None
            }


class Photo2CartoonOnxx:
    def __init__(self):
        self.pre = Preprocess()        
        assert os.path.exists('./models/photo2cartoon_weights.onnx'), "[Step1: load weights] Can not find 'photo2cartoon_weights.onnx' in folder 'models!!!'"
        self.session = onnxruntime.InferenceSession('./models/photo2cartoon_weights.onnx')
        print('[Step1: load weights] success!')

    def inference(self, img):
        # face alignment and segmentation
        face_rgba = self.pre.process(img)
        if face_rgba is None:
            print('[Step2: face detect] can not detect face!!!')
            return None
        
        print('[Step2: face detect] success!')
        face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
        face = face_rgba[:, :, :3].copy()
        mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
        face = (face*mask + (1-mask)*255) / 127.5 - 1

        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)

        # inference
        cartoon = self.session.run(['output'], input_feed={'input':face})

        # post-process
        cartoon = np.transpose(cartoon[0][0], (1, 2, 0))
        cartoon = (cartoon + 1) * 127.5
        cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
        cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        print('[Step3: photo to cartoon] success!')
        return cartoon

if __name__ == '__main__':
    app.run(host='127.0.0.1')