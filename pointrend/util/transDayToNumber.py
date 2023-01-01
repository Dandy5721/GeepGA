import pandas as pd


GTpath='孕周真实值(视频获得).csv'
ResultPath='MA_Result.csv'
df_GT=pd.DataFrame(pd.read_csv(GTpath,encoding="GBK"))
df_R=pd.DataFrame(pd.read_csv(ResultPath,encoding="GBK"))
Name=list(df_GT["Name"])
GTweek=list(df_GT['Week(视频)'])
Rweek=list(df_R['MA'])
dict={}

GTweekP=[0 for i in range(len(Name))]

for i in range(len(Name)):
    if not pd.isnull(GTweek[i]):
        w=float(GTweek[i].split("+")[0])
        d=float(GTweek[i].split("+")[1])/7.0
        GTweekP[i]=w+d
    else:
        GTweekP[i]=''
dict["Name"]=Name
dict['GT']=GTweekP
dict["Result"]=Rweek

df=pd.DataFrame(dict)

df.to_csv("finale.csv",index=0)