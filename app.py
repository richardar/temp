from flask import Flask,request
import shutil
import os
from retina_face import face_det, face_extract_r,half_body,blurface
from PIL import Image


app = Flask(__name__)


@app.route('/ver', methods=['POST'])
def verify():
    img1 = request.files['image1']
    img2 = request.files['image2']
    img1_pil = Image.open(img1)
    img2_pil= Image.open(img2)
    models={}
    
    shutil.rmtree("/home/quadro/facescore/TFace/recognition/AdaFace/face_alignment/test_images")
    os.makedirs("/home/quadro/facescore/TFace/recognition/AdaFace/face_alignment/test_images",exist_ok=True)
    img1_pil.save("/home/quadro/facescore/TFace/recognition/AdaFace/face_alignment/test_images/img1.jpg")
    img2_pil.save("/home/quadro/facescore/TFace/recognition/AdaFace/face_alignment/test_images/img2.jpg")
    models['Adaface']={"similarity_score":str(run()),
                       "model": "adaface_ir50_ms1mv2.ckpt"}
    
    
@app.route("/peoplehalf", methods=['POST'])
def peoplehalf():
        imgarr = Image.open(request.files["file"])
        return half_body(imgarr)