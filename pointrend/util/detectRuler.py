import os
from shutil import copyfile

path="C:\\Users\\aiyunji-xj\\Desktop\\肱骨"
filelist=os.listdir(path)

jsons=[]
for file in filelist:
    if file.endswith(".json"):
        jsons.append(file)
noruler=[]
copypath="C:\\Users\\aiyunji-xj\\Desktop\\肱骨noruler"

for json in jsons:

    with open(os.path.join(path,json),encoding="utf-8") as f:
        contents=f.read()
        # print(contents)
        if "Add-Ruler" not in contents:
            print(json)
            # print(json.split(".")[0])
            copyfile(os.path.join(path, json), os.path.join(copypath, json))
            copyfile(os.path.join(path, json.split(".")[0] + ".jpg"),
                     os.path.join(copypath, json.split(".")[0] + ".jpg"))

            noruler.append(json.split(".")[0])


