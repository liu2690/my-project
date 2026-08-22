import numpy as np
import os


def read_files(folder, include=[], exclude=[]):

    files=[]


    for f in os.listdir(folder):

        if not f.endswith(".dat"):
            continue

        if ":Zone.Identifier" in f:
            continue


        flag=True


        # 必须包含
        for key in include:
            if key not in f:
                flag=False


        # 排除
        for key in exclude:
            if key in f:
                flag=False


        if flag:
            files.append(f)


    files.sort()

    print("匹配数量:",len(files))

    for f in files[:10]:
        print(f)

    data=[]


    for f in files:

        print("读取:", f)

        signal=np.loadtxt(
            os.path.join(
                folder,
                f
            )
        )

        data.append(signal)


    return np.array(data), files