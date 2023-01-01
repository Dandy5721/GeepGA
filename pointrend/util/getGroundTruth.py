import os
import csv
import json
import math
import pandas as pd
def getDistance(p,q):
    return math.sqrt(math.pow(p[0]-q[0],2)+math.pow(p[1]-q[1],2))

def oval(label,shapes):
    axis=[]
    ppc=0
    for s in shapes:
        points = s["points"]
        # print(s["label"])
        # print(label)
        if s["label"]==label:
            axis.append(getDistance(points[0],points[1]))
        else:
            ruler=getDistance(points[0],points[1])
            ppc = ruler / float(s["label"])

    a = max(axis) / 2
    b = min(axis) / 2
    # print(b,a)
    # 椭圆周长公式 百度百科 公式8
    q = a + b
    h = math.pow(((a - b) / (a + b)), 2)
    m = 22 / 7 * math.pi - 1
    n = math.pow((a - b) / a, 33.697)
    L = math.pi * q * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h))) * (1 + m * n)/ppc
    # print("HC",HC)
    return L




def segment(label,shapes):
    ppc = 0
    L=0
    for s in shapes:
        points = s["points"]

        if s["label"] == label:
            L=getDistance(points[0], points[1])
        else:
            ruler = getDistance(points[0], points[1])
            ppc = ruler / float(s["label"])
    return L/ppc

path="C:\\Users\\aiyunji-xj\\Desktop\\newGT"
# path="C:\\Users\\aiyunji-xj\\Desktop\\gttest"
fileList=os.listdir(path)
namepath="C:\\Users\\aiyunji-xj\\Desktop\\GTName.csv"
title = ["Name", "BPD",
                 'HC',
                 'AC',
                 'FL']
df=pd.DataFrame(pd.read_csv(namepath))
print(df.columns)
Name=list(df['Name'])
BPD=[[] for i in range(len(Name))]
HC=[[] for i in range(len(Name))]
AC=[[] for i in range(len(Name))]
FL=[[] for i in range(len(Name))]
for file in fileList:

    if file.endswith("json"):
        sp=file.split("_")
        name=sp[1]+"_"+sp[2]
        if name in Name:
            index=Name.index(name)
            json_path=os.path.join(path,file)
            with open(json_path, encoding="utf-8") as fp:
                json_data = json.load(fp)
                shapes=json_data["shapes"]
                labels=[]
                for s in shapes:
                    labels.append(s["label"])
                if 'BPD' in labels:
                    BPD[index].append(segment('BPD',shapes))
                elif 'HC' in labels:

                    HC[index].append(oval('HC',shapes))
                elif 'AC' in labels:
                    AC[index].append(oval('AC',shapes))
                elif 'FL' in labels:
                    FL[index].append(segment('FL',shapes))
                else:
                    print("nothing")
                    print(file)

print(BPD)
print(HC)
print(AC)
print(FL)
for i in range(len(Name)):
    if len(BPD[i])!=0:
        BPD[i]=sum(BPD[i])/len(BPD[i])*10
    else:
        BPD[i]=''
    if len(HC[i])!=0:
        HC[i] = sum(HC[i]) / len(HC[i])*10
    else:
        HC[i]=''
    if len(AC[i])!=0:
        AC[i] = sum(AC[i]) / len(AC[i])*10
    else:
        AC[i]=''
    if len(FL[i])!=0:
        FL[i] = sum(FL[i]) / len(FL[i])*10
    else:
        FL[i]=''
dict={}
dict['Name']=Name
dict['BPD']=BPD
dict['HC']=HC
dict['AC']=AC
dict['FL']=FL
df=pd.DataFrame(dict)
df.to_csv("GroundTruth.csv",index=0)
