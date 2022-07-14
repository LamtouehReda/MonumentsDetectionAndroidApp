import numpy as np
import cv2
from PIL import Image
import base64
import io

def main(data):
    decoded_data=base64.b64decode(data)
    np_data=np.fromstring(decoded_data,np.uint8)
    img=cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    pil_img=Image.fromarray(img_gray)

    buff=io.BytesIO()
    pil_img.save(buff,format="PNG")
    img_str=base64.b64encode(buff.getvalue())
    return ""+str(img_str,'utf-8')
