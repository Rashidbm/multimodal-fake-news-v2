"""Optional image-only face localization; no target labels or annotated boxes are inputs."""
import json
from pathlib import Path

import cv2
import numpy as np

from .preprocess import sha256

ROOT=Path(__file__).resolve().parents[2]
DIRECTORY=ROOT/'external/yunet'


class FaceCrop:
    def __init__(self,context=1.6,max_side=640):
        source=json.loads((DIRECTORY/'source.json').read_text());model=DIRECTORY/'face_detection_yunet_2023mar.onnx'
        if sha256(model)!=source['sha256']:raise ValueError('Face locator weights changed')
        cv2.setNumThreads(1);self.context=context;self.max_side=max_side
        self.detector=cv2.FaceDetectorYN.create(str(model),'',(320,320),.8,.3,5000,cv2.dnn.DNN_BACKEND_OPENCV,cv2.dnn.DNN_TARGET_CPU)

    def locate(self,image):
        rgb=np.asarray(image.convert('RGB'));height,width=rgb.shape[:2]
        scale=min(1.,self.max_side/max(width,height));scaled=cv2.resize(rgb[:,:,::-1],(max(1,round(width*scale)),max(1,round(height*scale))))
        self.detector.setInputSize((scaled.shape[1],scaled.shape[0]));_,faces=self.detector.detect(scaled)
        if faces is None or not len(faces):return dict(box=[0,0,width,height],boxes=[[0,0,width,height]],detected=False,faces=0)
        faces=sorted(faces,key=lambda f:f[2]*f[3],reverse=True);boxes=[]
        for face in faces:
            x,y,w,h=face[:4]/scale;size=max(w,h)*self.context;cx=x+w/2;cy=y+h/2
            box=[max(0,int(np.floor(cx-size/2))),max(0,int(np.floor(cy-size/2))),
                 min(width,int(np.ceil(cx+size/2))),min(height,int(np.ceil(cy+size/2)))]
            if box[2]<=box[0] or box[3]<=box[1]:raise ValueError('Invalid detected crop')
            boxes.append(box)
        return dict(box=boxes[0],boxes=boxes,detected=True,faces=len(faces),confidence=float(faces[0][-1]))

    def __call__(self,image):return image.crop(self.locate(image)['box'])
