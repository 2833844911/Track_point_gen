
import torch
from torch import nn
import json
import numpy as np
from zBzier import bezierTrajectory


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('./gjsetting.txt',encoding='utf-8') as f:
    gjsett = json.loads(f.read())

class modelGj(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        x2, (_, l) = self.lstmEncode(x)
        alld = self.linDecode(x2)

        xd = alld[:, :, 0]
        yd = alld[:, :, 1]

        xd = self.sig(xd)
        yd = self.sig(yd)
        return xd, yd


def showGj(tar,out):
    print("长度:",len(tar), len(out))
    import matplotlib.pyplot as plt
    x1 = np.array(tar)[:,0]
    y1 = np.array(tar)[:, 1]
    plt.scatter(x1, y1, c='r', s=10)

    x2 = np.array(out)[:, 0]
    y2 = np.array(out)[:, 1]
    plt.scatter(x2, y2, c='b', s=10)

    plt.show()

def getGj(data):
    data = data.copy()
    cd = len(data)
    data += [[0,0]]*(maxNum-cd)
    data = np.array(data, dtype=np.double)
    mx = np.min(data[:,0])
    my = np.min(data[:,1])
    data[:cd,0] = ((data[:cd,0] - mx)/  gjsett['x'])
    data[:cd, 1] = ((data[:cd,1] - my)/ gjsett['y'])
    dm = data.tolist()

    d = torch.tensor(dm, dtype=torch.float32).to(device).unsqueeze(0)
    st = 1
    for _ in range(d.shape[1] // 20):
        d[:, st:(st + 20), :] = 0
        st = st + 20 + 1
        if st >= d.shape[1]:
            break
    if d.shape[1] // 20 < d.shape[1] / 20:
        d[:, st:-1, :] = 0

    outx, outy = mod(d)
    outx = outx.squeeze(0)
    outy = outy.squeeze(0)
    outx = outx.tolist()
    outy = outy.tolist()
    outdata = []

    for i in range(len(outx)):
        if i == 0:
            outdata.append([dm[0][0]* gjsett['x'] +mx,dm[0][1]*gjsett['y']+my  ])
            continue
        if i == len(outx)-1:
            outdata.append([dm[i][0] * gjsett['x'] + mx, dm[i][1] * gjsett['y'] + my])
            continue
        outdata.append([outx[i] * gjsett['x'] +mx, outy[i]*gjsett['y']+my ])
        if len(outdata) == cd:
            return outdata

    return outdata

mod = torch.load('./modelyzm.pth')

maxNum = 90

if __name__ == '__main__':
    # dataList = json.load(open('./gj.json', encoding='utf-8'))
    # data = []
    # start =1490
    # for i in range(start, start +90):
    #     data.append([dataList[i]['x'], dataList[i]['y']])


    # 生成一段轨迹 然后让ai修改 但是模型输入点需要为90个点
    bzr = bezierTrajectory()
    hh = bzr.trackArray([100,200],[950,700],90,6,3,0.4)
    data = hh['trackArray'].tolist()


    out = getGj(data)
    showGj(data, out)




