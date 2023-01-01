import os
import cv2
import json
from labelme import PY2
from labelme import QT4
from labelme.logger import logger
import base64
import io
import numpy as np
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
import os
import os.path as osp
import json



def apply_exif_orientation(image):
    try:
        exif = image._getexif()
    except AttributeError:
        exif = None

    if exif is None:
        return image

    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in exif.items()
        if k in PIL.ExifTags.TAGS
    }

    orientation = exif.get('Orientation', None)

    if orientation == 1:
        # do nothing
        return image
    elif orientation == 2:
        # left-to-right mirror
        return PIL.ImageOps.mirror(image)
    elif orientation == 3:
        # rotate 180
        return image.transpose(PIL.Image.ROTATE_180)
    elif orientation == 4:
        # top-to-bottom mirror
        return PIL.ImageOps.flip(image)
    elif orientation == 5:
        # top-to-left mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
    elif orientation == 6:
        # rotate 270
        return image.transpose(PIL.Image.ROTATE_270)
    elif orientation == 7:
        # top-to-right mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
    elif orientation == 8:
        # rotate 90
        return image.transpose(PIL.Image.ROTATE_90)
    else:
        return image


# @staticmethod
def load_image_file(filename):
    try:
        image_pil = PIL.Image.open(filename)
    except IOError:
        logger.error('Failed opening image file: {}'.format(filename))
        return

    # apply orientation to image according to exif
    image_pil = apply_exif_orientation(image_pil)

    with io.BytesIO() as f:
        ext = osp.splitext(filename)[1].lower()
        if PY2 and QT4:
            format = 'PNG'
        elif ext in ['.jpg', '.jpeg']:
            format = 'JPEG'
        else:
            format = 'PNG'
        image_pil.save(f, format=format)
        f.seek(0)
        return f.read()

path="C:\\Users\\aiyunji-xj\\Desktop\\doubleGT"
savepath="C:\\Users\\aiyunji-xj\\Desktop\\cutdouble"
filelist=os.listdir(path)
for file in filelist:
    if file.endswith("jpg"):
        print(file)
        img_path=os.path.join(path,file)
        img=cv2.imread(img_path)
        w=img.shape[1]
        boundary=int(w*0.5)
        # 处理json
        json_path=img_path.replace("jpg","json")
        with open(json_path, encoding="utf-8") as fp:
            json_data = json.load(fp)
            shapes = json_data["shapes"]
            left=1601
            right=-1
            error=50
            for shape in shapes:
                points=shape["points"]
                label=shape['label']
                if label=="BPD":
                    point=points[0]
                    if point[0]>boundary:
                        left=800+error
                        right=1600-error
                    else:
                        left=0+error
                        right=800-error
                    break

                else:
                    for p in points:

                        if p[0]<left:
                            left=p[0]
                        elif p[0]>right:
                            right=p[0]

            left=int(max(left-error,0))
            right=int(min(right+error,w))

            for shape in shapes:
                points=shape["points"]
                for p in points:
                    p[0]-=left

            print("w",left,right)
            new_img=img[:,left:right,:]

            img_save = os.path.join(savepath, img_path.split("\\")[-1])
            cv2.imwrite(img_save, new_img)

            imgData=load_image_file(img_save)
            imgData=base64.b64encode(imgData).decode("gbk")
            str1=imgData


            json_data['imageData']=str1
            json_save=os.path.join(savepath,json_path.split("\\")[-1])
            json.dump(json_data,open(json_save,"w"))
        # 处理双幅图
        # break


