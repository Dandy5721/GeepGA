import pandas as pd
df=pd.DataFrame(pd.read_csv("gtAndResult.csv"))

Name=list(df['Name'])
title=["BPD",'HC','AC','FL']
Error={}
Error['BPD_E']=['' for i in range(len(Name))]
Error['HC_E']=['' for i in range(len(Name))]
Error['AC_E']=['' for i in range(len(Name))]
Error['FL_E']=['' for i in range(len(Name))]
for t in title:

    E=Error[t+"_E"]
    GT=list(df[t+"_GT"])
    R = list(df[t + "_R"])
    for j in range(len(Name)):
        if not pd.isnull(GT[j]):
            if pd.isnull(R[j]):
                E[j]=GT[j]
            else:
                E[j]=abs(GT[j]-R[j])

dict={}
dict['Name']=Name

for t in title:
    dict[t+'_GT']=df[t+'_GT']
    dict[t+'_R']=df[t+'_R']
    dict[t+'_E']=Error[t+'_E']
dfs=pd.DataFrame(dict)
dfs.to_csv("Error.csv",index=0)
