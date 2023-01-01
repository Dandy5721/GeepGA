import cv2
import numpy as np
import math
def getPPC(img,box):
    box=[int(x) for x in box]
    # 先h 后 w
    ruler_img=img[box[1]:box[3],box[0]:box[2],:]
    # cv2.imshow("ruler",ruler_img)
    # cv2.waitKey(0)

    gray = cv2.cvtColor(ruler_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray",gray)
    # cv2.waitKey(0)
    light=[i for j in gray for i in j]
    light.sort(reverse=True)
    scale = 0.02
    threshold = light[int(len(light) * scale)]
    # print(threshold)
    if threshold<50:
        threshold=60
    elif threshold>170:
        threshold=122
    elif threshold>150:
        threshold=110
    elif threshold<=110:
        pass
    else:
        threshold=124

    # print("threshold",threshold)

    # print(threshold)
    ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # 找出尺子内的所有刻度
    # print(len(contours))
    # cv2.drawContours(ruler_img,contours,-1,(0,0,255),1)
    # cv2.imshow("haha",ruler_img)
    # cv2.waitKey(0)
    #按左上角的第一个点的y排序
    contours.sort(key=lambda y: y[0, 0, 1])

    # print(contours[0])
    # 过滤掉大背景
    contours1 = []
    for contour in contours:
        if 1 < len(contour) < 30:
            contours1.append(contour)
    contours2 = []
    # 选出一条过轮廓最多的y
    # 过滤掉边缘小背景
    for i in range(box[2] - box[0]):
        temp = []
        for contour in contours1:
            # print(contour)
            xs = contour[:, :, 0]
            xs = xs.reshape(1, -1)
            #一个轮廓所有点的横坐标
            xs = xs[0]
            if i in xs:
                temp.append(contour)
        if len(temp) > len(contours2):
            contours2 = temp
    # cv2.drawContours(ruler_img, contours2, -1, (0, 0, 255), 1)
    # cv2.imshow("ruler1", ruler_img)
    # cv2.waitKey(0)
    # 得出两个刻度尺之间准确的像素个数 这个已经十分准确
    if len(contours2) < 2:
        print("轮廓数小于2")
        pixel_per_scale = 0
        return pixel_per_scale
    else:
        dict = {}
        for i in range(len(contours2) - 1):
            temp = abs(contours2[i][0][0][1] - contours2[i + 1][0][0][1])
            if temp < 10:
                continue
            if temp in dict.keys():
                dict[temp] = dict[temp] + 1
            else:
                dict[temp] = 1
        # print(dict)
        #构建约等于dict
        newDict={}
        for key1 in dict.keys():
            newDict[key1]=dict[key1]
            for key2 in dict.keys():
                if abs(key1-key2)<3:
                    newDict[key1]+=1

        max = 0
        maxkey = 10
        for key in dict.keys():
            if dict[key] > max:
                maxkey = key
                max = dict[key]

        # print(newDict)
        # print("分析得出", maxkey)
        pixel_per_scale = maxkey
        # 过滤掉尺子内部小背景
        # print(pixel_per_scale)
        # 外接矩形  修改
        recs = []
        for contour in contours2:
            rec = cv2.boundingRect(contour)
            recs.append(rec)

        # cv2.drawContours(rulerimg, contours2, -1, (0, 0, 255), 1)
        # print(recs)
        recs1 = []
        # 不依赖第一个刻度的正确识别
        # 遍历所有矩阵，并剪枝
        for i in range(len(recs) -1):
            for j in range( i+1,len(recs)):
                # print(recs[j],recs[i])
                if abs(recs[j][1] - recs[i][1]) > pixel_per_scale + 10:
                    break
                if pixel_per_scale - 2 <= recs[j][1] - recs[i][1] <= pixel_per_scale + 2:
                    if recs[i] not in recs1:
                        recs1.append(recs[i])
                        # cv2.rectangle(ruler_img, (recs[i][0], recs[i][1]),
                        #               (recs[i][0] + recs[i][2], recs[i][1] + recs[i][3]),
                        #               (0, 255, 255), 1)
                    if recs[j] not in recs1:
                        recs1.append(recs[j])
                        # cv2.rectangle(ruler_img, (recs[j][0], recs[j][1]),
                        #               (recs[j][0] + recs[j][2], recs[j][1] + recs[j][3]),
                        #               (0, 255, 255), 1)
    # print(recs1)
    # cv2.imshow("recs",ruler_img)
    # cv2.waitKey(0)
    # print("过滤前轮廓个数", len(contours))
    # # print(contours)
    # print("过滤后轮廓个数", len(recs1))
    count1 = 0

    last2Rec = (0, 0, 0, 0)
    lastRec = (0, 0, 0, 0)
    # 判断种类 默认3刻度尺
    kind = 3
    for rec in recs1:

        # 符合大小大顺序的是0.5刻度尺
        if  last2Rec[2] == rec[2] and pixel_per_scale - 2 < abs(rec[1] - last2Rec[1]) / 2 < pixel_per_scale + 2:
            if lastRec[2] < rec[2]:
                kind = 0.5
                break

        if lastRec[2] == rec[2] and pixel_per_scale - 2 < abs(rec[1] - lastRec[1]) < pixel_per_scale + 2:
            count1 += 1
        # 连续2个刻度一样就是1刻度尺
        if count1 == 1:
            kind = 1
            break
        last2Rec = lastRec
        lastRec = rec
    if kind == 0.5:
        pixel_per_scale *= 2
    # print(kind)
    # color = (0, 255, 0)
    # cv2.line(img, (box[0], box[1]), (box[2] + 10, box[1]), color)
    # cv2.line(img, (box[0], box[1]), (box[0], box[3]), color)
    # cv2.line(img, (box[0], box[3]), (box[2] + 10, box[3]), color)
    # cv2.line(img, (box[2] + 10, box[1]), (box[2] + 10, box[3]), color)
    # cv2.imshow("ruler",img)
    # cv2.waitKey(0)
    # print("ppc in this image is ", pixel_per_scale)
    # s = str(pixel_per_scale)
    # cv2.putText(rulerimg,"ppc:",(0,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA)
    # cv2.putText(rulerimg, s, (2, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    # cv2.imshow("final", rulerimg)
    # cv2.waitKey()
    return pixel_per_scale

def getRotateAngle(img, mask):
    ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # temp=mask.copy()
    # print(contours)
    # cv2.drawContours(temp,contours,-1,(0,255,255),4)
    # cv2.imshow("before rotate",temp)
    # cv2.waitKey(0)
    contours.sort(key=lambda c: len(c), reverse=True)
    contours = contours[0]
    rect = cv2.minAreaRect(contours)
    angle = rect[2]
    w, h = rect[1]
    # print("w:", w, "h:", h)
    # print(angle)
    if w < h:
        angle = 90 + angle
    # print(angle)

    box = cv2.boxPoints(rect)
    img2 = np.zeros_like(img)
    img2[:, :, 0] = binary
    img2[:, :, 1] = binary
    img2[:, :, 2] = binary
    color = (0, 255, 0)
    cv2.line(img2, (box[0][0], box[0][1]), (box[1][0], box[1][1]), color,5)
    cv2.line(img2, (box[1][0], box[1][1]), (box[2][0], box[2][1]), color,5)
    cv2.line(img2, (box[2][0], box[2][1]), (box[3][0], box[3][1]), color,5)
    cv2.line(img2, (box[3][0], box[3][1]), (box[0][0], box[0][1]), color,5)
    # cv2.imshow("rect",img2)
    cv2.imwrite("C:\\Users\\aiyunji-xj\\Desktop\\figure\\figure4\\FL_rect.png",img2)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return angle


def calculateRotateWidth(out_rotation, in_rotation):
    out_contours,h=cv2.findContours(out_rotation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    in_contours,h=cv2.findContours(in_rotation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out_contours.sort(key=lambda c:len(c),reverse=True)
    in_contours.sort(key=lambda c:len(c),reverse=True)

    # cv2.imshow("out",in_rotation)
    # cv2.waitKey(0)

    out_contour=out_contours[0]
    in_contour=in_contours[0]
    #按x从左到右每个点排序
    out_contour=sorted(out_contour, key=lambda x: x[0][0], reverse=False)
    in_contour=sorted(in_contour, key=lambda x: x[0][0], reverse=False)

    start = in_contour[0][0][0]
    end = in_contour[-1][0][0]
    temp=[]
    for point in out_contour:
        if point[0][0]>=start and point[0][0]<end+1:
            temp.append(point)
    out_contour=temp
    maxWidth = 0
    pointY=[]

    p = 0
    q = 0
    for i in range(start,end+1):
        out_points=[]
        in_points=[]

        for j in range(p,len(out_contour)):
            if out_contour[j][0][0] == i:
                out_points.append([out_contour[j][0][0], out_contour[j][0][1]])
            else:
                p=j
                break

        for k in range(q,len(in_contour)):
            if in_contour[k][0][0] == i:
                in_points.append([in_contour[k][0][0], in_contour[k][0][1]])
            else:
                q=k
                break
        #按y轴从小到大排序
        out_points=sorted(out_points,key=lambda x:x[1],reverse=False)
        in_points=sorted(in_points,key=lambda x:x[1],reverse=False)
        # print(out_points)
        # print(in_points)
        if len(out_points)==0 or len(in_points)==0:
            # print("hahaa")
            return []
        width=in_points[-1][1]-out_points[0][1]
        if width>maxWidth:
            maxWidth=width
            pointY=[]
            pointY.append(out_points[0])
            pointY.append(in_points[-1])
    return pointY

def getBPD(img,out_mask,in_mask):
    #获得outskull的外接圆
    ret, out_binary = cv2.threshold(out_mask, 0, 255, cv2.THRESH_BINARY)
    # cv2.imshow("out",out_binary)
    # cv2.imwrite("C:\\Users\\aiyunji-xj\\Desktop\\figure\\figure4\\BPD_out_before.png",out_binary)

    # cv2.waitKey(0)
    contours, h = cv2.findContours(out_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img,contours,-1,(0,255,0),5)

    ret, in_binary = cv2.threshold(in_mask, 0, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("C:\\Users\\aiyunji-xj\\Desktop\\figure\\figure4\\BPD_in_before.png",in_binary)

    contours2, h = cv2.findContours(in_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours2, -1, (0, 255, 255), 5)
    # cv2.imshow("img", img)
    cv2.imwrite("C:\\Users\\aiyunji-xj\\Desktop\\figure\\figure4\\BPD_contours_before.png",img)
    # cv2.waitKey(0)
    contours.sort(key=lambda c:len(c),reverse=True)

    contours = contours[0]
    # print(len(contours))
    ellipse = cv2.fitEllipse(contours)
    #获得outskull椭圆

    angle=ellipse[2]-90
    # print(angle)
    print("angle",angle)


    ret, in_binary=cv2.threshold(in_mask, 0, 255, cv2.THRESH_BINARY)
    height, width = out_binary.shape[:2]

    mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    img_rotation = cv2.warpAffine(img.copy(), mat_rotation, (width, height), borderValue=(0, 0, 0))

    out_rotation = cv2.warpAffine(out_binary.copy(), mat_rotation, (width, height), borderValue=(0, 0, 0))
    in_rotation=cv2.warpAffine(in_binary.copy(), mat_rotation, (width, height), borderValue=(0, 0, 0))
    # cv2.imshow("img,rotate",img_rotation)
    cv2.imwrite("C:\\Users\\aiyunji-xj\\Desktop\\figure\\figure4\\BPD_contours_after.png",img_rotation)

    cv2.waitKey(0)
    img2 = np.zeros_like(img)
    img2[:, :, 0] = out_binary
    img2[:, :, 1] = out_binary
    img2[:, :, 2] = out_binary
    cv2.ellipse(img2,ellipse,(0,255,128),8)

    # cv2.imshow("before",img2)
    cv2.imwrite("C:\\Users\\aiyunji-xj\\Desktop\\figure\\figure4\\BPD_ell_before.png",img2)

    # img3 = np.zeros_like(img)
    # img3[:, :, 0] = out_rotation
    # img3[:, :, 1] = out_rotation
    # img3[:, :, 2] = out_rotation
    # cv2.imshow("after",out_rotation)
    # cv2.imwrite("C:\\Users\\aiyunji-xj\\Desktop\\figure\\figure4\\BPD_out_after.png",out_rotation)
    # cv2.imwrite("C:\\Users\\aiyunji-xj\\Desktop\\figure\\figure4\\BPD_in_after.png",in_rotation)

    # cv2.waitKey(0)
    pointY=calculateRotateWidth(out_rotation,in_rotation)
    # cv2.line(img_rotation, (pointY[0][0], pointY[0][1]), (pointY[1][0], pointY[1][1]), (0,0,255),5)
    # cv2.imshow("imr",img_rotation)
    # cv2.imwrite("C:\\Users\\aiyunji-xj\\Desktop\\figure\\figure4\\BPD_point_after.png",img_rotation)

    # cv2.waitKey(0)
    if len(pointY)==0:
        return -1
    # print(pointY)
    #把得到的pointY转换回原图
    angle=-angle
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    newY=[]
    for point in pointY:
        a = np.array([point[0], point[1], 1])
        b = np.dot(matRotation, a)
        newY.append([int(b[0]), int(b[1])])
    # print(newY)
    # cv2.line(img, (newY[0][0], newY[0][1]), (newY[1][0], newY[1][1]), (0, 0, 255), 5)
    # cv2.imshow("imr", img)
    # cv2.imwrite("C:\\Users\\aiyunji-xj\\Desktop\\figure\\figure4\\BPD_point_before.png",img)

    # cv2.waitKey(0)
    width = np.sqrt(pow(newY[0][0] - newY[1][0], 2) + pow(newY[0][1] - newY[1][1], 2))
    # print(newY)
    return width

def getHC(img,mask):
    ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, h = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours.sort(key=lambda c:len(c),reverse=True)
    img2=np.zeros_like(img)
    img2[:,:,0]=binary
    img2[:,:,1]=binary
    img2[:,:,2]=binary
    # cv2.imshow("3cmask",img2)
    contours=contours[0]
    ellipse=cv2.fitEllipse(contours)
    cv2.ellipse(img,ellipse,(0,255,0),5)
    cv2.ellipse(img2,ellipse,(0,255,0),5)

    # cv2.imshow("ellipse",img2)
    # cv2.imwrite("C:\\Users\\aiyunji-xj\\Desktop\\figure\\figure4\\HC_result.png",img)
    # cv2.imwrite("C:\\Users\\aiyunji-xj\\Desktop\\figure\\figure4\\HC_mask.png",binary)
    # cv2.imwrite("C:\\Users\\aiyunji-xj\\Desktop\\figure\\figure4\\HC_ell.png",img2)
    #
    # cv2.waitKey(0)
    #ellipse (center(x,y),(2b,2a),angle)
    size=ellipse[1]
    #size必定 （短轴，长轴）=（2b,2a）
    a=size[1]/2
    b=size[0]/2
    # print(b,a)
    #椭圆周长公式 百度百科 公式8
    q=a+b
    h=math.pow(((a-b)/(a+b)),2)
    m=22/7*math.pi-1
    n=math.pow((a-b)/a,33.697)
    HC=math.pi*q*(1+3*h/(10+math.sqrt(4-3*h)))*(1+m*n)
    # print("HC",HC)
    return HC

def getAC(img,mask):
    return getHC(img,mask)

def calculateRotateLength(img,rotation):
    contours, h = cv2.findContours(rotation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours.sort(key=lambda c:len(c),reverse=True)

    contours = contours[0]

    rect = cv2.minAreaRect(contours)
    # box = cv2.boxPoints(rect)
    # img2 = np.zeros_like(img)
    # img2[:, :, 0] = rotation
    # img2[:, :, 1] = rotation
    # img2[:, :, 2] = rotation
    # color = (0, 255, 0)
    # cv2.line(img2, (box[0][0], box[0][1]), (box[1][0], box[1][1]), color,5)
    # cv2.line(img2, (box[1][0], box[1][1]), (box[2][0], box[2][1]), color,5)
    # cv2.line(img2, (box[2][0], box[2][1]), (box[3][0], box[3][1]), color,5)
    # cv2.line(img2, (box[3][0], box[3][1]), (box[0][0], box[0][1]), color,5)
    # cv2.imshow("rect", img2)
    # cv2.imwrite("C:\\Users\\aiyunji-xj\\Desktop\\figure\\figure4\\FL_rect2.png", img2)

    # cv2.waitKey(0)

    contours = sorted(contours, key=lambda x: x[0][0], reverse=False)
    start = contours[0][0][0]
    end = contours[-1][0][0]
    # print(start, end)
    p = 0
    maxWidth = 0
    pointY = None
    for i in range(start, end + 1):
        points = []
        for j in range(p, len(contours)):
            if contours[j][0][0] == i:
                points.append([contours[j][0][0], contours[j][0][1]])
            else:
                p = j
                break

        points = sorted(points, key=lambda x: x[1], reverse=False)
        width = points[-1][1] - points[0][1]

        if width > maxWidth:
            maxWidth = width
            pointY = []
            pointY.append(points[0])
            pointY.append(points[-1])

    # cv2.line(img2, (pointY[0][0], pointY[0][1]), (pointY[1][0], pointY[1][1]), (0,0,255),4)
    # cv2.imshow("rect", img2)
    # cv2.imwrite("C:\\Users\\aiyunji-xj\\Desktop\\figure\\figure4\\FL_point.png", img2)
    #
    # cv2.waitKey(0)

    return pointY

def getFL(img,mask):

    ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    # cv2.imshow("before",mask)

    angle = getRotateAngle(img, binary)+90
    height, width = binary.shape[:2]
    # print(angle)
    mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    # cv2.imshow("before",binary)
    # cv2.imwrite("C:\\Users\\aiyunji-xj\\Desktop\\figure\\figure4\\FL_before.png",binary)
    rotation = cv2.warpAffine(binary.copy(), mat_rotation, (width, height), borderValue=(0, 0, 0))
    # cv2.imshow("after",rotation)
    # cv2.imwrite("C:\\Users\\aiyunji-xj\\Desktop\\figure\\figure4\\FL_after.png",rotation)
    pointsY = calculateRotateLength(img,rotation)
    if len(pointsY)<2:
        return -1
    angle=-angle
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    newY = []

    for point in pointsY:
        a = np.array([point[0], point[1], 1])
        b = np.dot(matRotation, a)
        newY.append([int(b[0]), int(b[1])])
    # print(newY)
    # cv2.line(img, (newY[0][0], newY[0][1]), (newY[1][0], newY[1][1]), (0,0,255),4)
    # cv2.imshow("FL_line",img)
    # cv2.imwrite("C:\\Users\\aiyunji-xj\\Desktop\\figure\\figure4\\FL_line.png",img)
    #
    # cv2.waitKey(0)

    img2 = np.zeros_like(img)
    img2[:, :, 0] = binary
    img2[:, :, 1] = binary
    img2[:, :, 2] = binary
    cv2.line(img2, (newY[0][0], newY[0][1]), (newY[1][0], newY[1][1]), (0,0,255),2)
    # cv2.imshow("FL_line",img2)
    # cv2.imwrite("C:\\Users\\aiyunji-xj\\Desktop\\figure\\figure4\\FL_point2.png",img2)

    # cv2.waitKey(0)
    length = np.sqrt(pow(newY[0][0] - newY[1][0], 2) + pow(newY[0][1] - newY[1][1], 2))
    # print(length)
    return length

def getHL(img,mask):
    return getFL(img,mask)
