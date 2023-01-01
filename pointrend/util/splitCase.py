import os

from shutil import copyfile
GT_path='C:\\Users\\aiyunji-xj\\Desktop\\newGT'
with open('C:\\Users\\aiyunji-xj\\Desktop\\problemImage.txt','r') as f:
    data=f.readlines()
print(data)
lines=[]
for line in data:
    if line!="":
        l=line.replace("\n","")
        lines.append(l)

fileList=os.listdir(GT_path)
print(len(lines))
save_path="C:\\Users\\aiyunji-xj\\Desktop\\case"
for file in fileList:
    if file.endswith("jpg"):
        sp=file.split("_")
        name=sp[1]+"_"+sp[2]
        if name in lines:
            # lines.remove(name)
            jpg_path=os.path.join(GT_path,file)
            new_path=os.path.join(save_path,name)
            isExists=os.path.exists(new_path)

            if not isExists:
                os.makedirs(new_path)


            copyfile(jpg_path,os.path.join(new_path,file))
            copyfile(jpg_path.replace("jpg","json"),(os.path.join(new_path,file)).replace("jpg","json"))

print(lines)