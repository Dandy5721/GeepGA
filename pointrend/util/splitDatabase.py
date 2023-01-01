import os
from shutil import copyfile

path="C:\\Users\\aiyunji-xj\\Desktop\\股骨"
filelist=os.listdir(path)

train=0.9

jpgs=[]

for file in filelist:
    if file.endswith("jpg"):
        jpgs.append(file)


print(len(jpgs))
trainPath="C:\\Users\\aiyunji-xj\\Desktop\\train"
valPath="C:\\Users\\aiyunji-xj\\Desktop\\val"
testPath="C:\\Users\\aiyunji-xj\\Desktop\\test"
totalSize=int(len(filelist)/2)

for i in range(totalSize):
    print(jpgs[i])
    if i<train*totalSize:
        copyfile(os.path.join(path,jpgs[i]),os.path.join(trainPath,jpgs[i]))
        copyfile(os.path.join(path,jpgs[i].split(".")[0]+".json"),os.path.join(trainPath,jpgs[i].split(".")[0]+".json"))

    elif i >= train*totalSize and i< (train+0.1)*totalSize:
        copyfile(os.path.join(path,jpgs[i]),os.path.join(valPath,jpgs[i]))
        copyfile(os.path.join(path,jpgs[i].split(".")[0]+".json"),os.path.join(valPath,jpgs[i].split(".")[0]+".json"))
    # else:
    #     copyfile(os.path.join(path, jpgs[i]), os.path.join(testPath, jpgs[i]))
    #     copyfile(os.path.join(path, jpgs[i].split(".")[0] + ".json"),
    #              os.path.join(testPath, jpgs[i].split(".")[0] + ".json"))

print("DONE!")
exit()