import os
import pandas as pd



GTpath="C:\\Users\\aiyunji-xj\\detectron2\\projects\\PointRend\\util\\GroundTruth.csv"
Resultpath="C:\\Users\\aiyunji-xj\\detectron2\\projects\\PointRend\\util\\Result.csv"
df_gt=pd.DataFrame(pd.read_csv(GTpath))
df_r=pd.DataFrame(pd.read_csv(Resultpath))
dict={}
for n in df_gt.columns:
    if n=='Name':
        dict[n]=df_gt[n]
    else:
        df_gt[n]=df_gt[n]
        dict[n+'_GT']=df_gt[n]
        dict[n+'_R']=df_r[n]

df=pd.DataFrame(dict)

df.to_csv("gtAndResult.csv",index=0)