import csv
import pandas as pd
def getName():
    name_path="C:\\Users\\aiyunji-xj\\Desktop\\GTName.csv"
    df = pd.DataFrame(pd.read_csv(name_path))

    return list(df["Name"])

name=getName()
print(name)
week=["" for i in range(len(name))]
f=open("C:\\Users\\aiyunji-xj\\Desktop\\606小时病例结果集合_result-final.csv","r",encoding='gbk')
row=csv.reader(f)
j=0
arr=[]
for r in row:
    n=r[14].split(".")[0]
    if n in name:
        i=name.index(n)
        if i in arr:
            print(n)
        arr.append(i)
        week[i]=r[11]
        j+=1
arr.sort()
print(arr)
dict={}

dict["Name"]=name
dict["Week"]=week
save=pd.DataFrame(dict)
save.to_csv("PregnancyWeek.csv",index=0)
