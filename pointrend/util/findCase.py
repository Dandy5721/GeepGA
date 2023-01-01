import pandas as pd

df= pd.DataFrame(pd.read_csv("C:\\Users\\aiyunji-xj\\detectron2\\projects\\PointRend\\util\\gtAndResult.csv"))

txt=open("C:\\Users\\aiyunji-xj\\Desktop\\problemImage.txt","w")
Name=list(df['Name'])
BPD_GT=list(df['BPD_GT'])
BPD_R=list(df['BPD_R'])
HC_GT=list(df['HC_GT'])
HC_R=list(df['HC_R'])
AC_GT=list(df['AC_GT'])
AC_R=list(df['AC_R'])
FL_GT=list(df['FL_GT'])
FL_R=list(df['FL_R'])



for i in range(len(Name)):

    if not pd.isnull(BPD_GT[i]):
        if abs(BPD_GT[i]-BPD_R[i])>15:
            txt.write(Name[i]+"\n")
            continue
    if not pd.isnull(HC_GT[i]):
        if abs(HC_GT[i]-HC_R[i])>15:
            txt.write(Name[i]+"\n")
            continue
    if not pd.isnull(AC_GT[i]):
        if abs(AC_GT[i]-AC_R[i])>15:
            txt.write(Name[i]+"\n")
            continue
    if not pd.isnull(FL_GT[i]):
        if abs(FL_GT[i]-FL_R[i])>15:
            txt.write(Name[i]+"\n")
            continue
txt.close()