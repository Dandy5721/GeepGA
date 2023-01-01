import pandas as pd
import os
dict={}
dict['Name']=set()

GT_path='C:\\Users\\aiyunji-xj\\Desktop\\20200831_测量值重新审核'
fl=os.listdir(GT_path)
print(len(fl))
for f in fl:
    if f.endswith("jpg"):
        print(f)
        sp=f.split("_")
        name=sp[1]+'_'+sp[2]
        dict["Name"].add(name)
l=dict['Name']

arr=list(l)

dict['Name']=arr
df=pd.DataFrame(dict)
df.to_csv("C:\\Users\\aiyunji-xj\\Desktop\\GTName.csv",index=0)