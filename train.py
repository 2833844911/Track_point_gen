import torch
from torch import nn
import json
import random
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataList = json.load(open('./gj2.json', encoding='utf-8'))

# 设置训练生成轨迹点的个数范围
fawei = [30, 150]



maxNum = fawei[1]+1

def getmax(data):
    maxX = 0
    maxY = 0

    for i in range(len(data)-maxNum):
        fd = []
        for f in range(maxNum):
            fd.append([data[i+f]['x'], data[i+f]['y']])
        fd = np.array(fd)

        maxXDx = np.max(fd[:,0])
        maxXDy = np.max(fd[:, 1])

        minXDx = np.min(fd[:,0])
        minXDy = np.min(fd[:, 1])
        maxXDx = maxXDx- minXDx
        maxXDy = maxXDy- minXDy

        if maxXDx > maxX:
            maxX = maxXDx
        if maxXDy > maxY:
            maxY = maxXDy
    return maxX, maxY

allMx, allMy = getmax(dataList)
print("最大xy",allMx, allMy)
with open('./gjsetting.txt', 'w', encoding='utf-8') as f:
    f.write(json.dumps({"x":int(allMx), "y":int(allMy)}))

def getDian(data):
    cd = len(data)
    # data += [[0,0]]*(maxNum-cd)
    data = np.array(data, dtype=np.double)
    mx = np.min(data[:,0])
    my = np.min(data[:,1])
    data[:cd,0] = ((data[:cd,0] - mx)/ allMx)
    data[:cd, 1] = ((data[:cd,1] - my)/ allMy)
    return data.tolist()


class modelGj(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstmEncode = nn.LSTM(2,50,num_layers=2, batch_first=True, bidirectional=True)


        self.linDecode = nn.Sequential(
            nn.Linear(100, 50),
            nn.Linear(50, 50),
            nn.Linear(50,2),
        )

        self.sig = nn.Sigmoid()



    def forward(self,x):
        x = x.clone()

        j = []
        st = 1
        buc = 2
        step = x.shape[1] // buc
        for _ in range(x.shape[1]//step):
            j.append(st-1)
            x[:,st:(st+step),:] = 0
            st = st+step+1
            if st+ step >= x.shape[1]:
                break
        if x.shape[1]//step < x.shape[1]/step:
            j.append(x.shape[1]-1)
            x[:, st:-1, :] = 0


        x2, (_,l) = self.lstmEncode(x)
        alld = self.linDecode(x2)

        xd = alld[:,:,0]
        yd = alld[:,:,1]

        xd = self.sig(xd)
        yd = self.sig(yd)
        return xd, yd, j

class dataLoad(Dataset):
    def __init__(self, data):
        super().__init__()
        self.length = len(data) - maxNum
        self.dataList = data

    def __getitem__(self, item):
        data = []
        for i in range(item,item+ random.randint(fawei[0], fawei[1])):
            data.append([self.dataList[i]['x'], self.dataList[i]['y']])
        d = getDian(data)
        data = d

        return torch.tensor(data,dtype=torch.float32).to(device)
    def __len__(self):
        return self.length


class myLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.bzloss = nn.MSELoss()
    def forward(self,outx, outy, tar, j):
        # xy坐标差距 使预测点的位置更加准确
        ls = torch.mean(torch.abs(outx - tar[:,:,0]))
        ls2 = torch.mean(torch.abs(outy - tar[:,:,1]))

        # 特点位置点的xy差距 使模型预测的结果需要更加接近这些点
        sls = torch.mean(torch.abs(outx[:, j] - tar[:, j, 0]))
        sls2 = torch.mean(torch.abs(outy[:, j] - tar[:, j, 1]))

        # xy坐标方差差距 使预测点的分布更加准确
        fanc = torch.abs(torch.var(outx) - torch.var(tar[:,:,0])) +torch.abs(torch.var(outy) - torch.var(tar[:,:,1]))
        loss = (ls+ls2 + 0.8*fanc + (sls+sls2) * 8)*10
        return loss, fanc, ls+ls2, sls+sls2

dm = dataLoad(dataList)

# 训练多少epoch
epochs = 200
try:
    model = torch.load('./modelyzm.pth')
except:
    model = modelGj()
model.to(device)

# 定义损失函数和优化器
criterion = myLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
max_xl=4000
for epoch in range(epochs):
    dataTrain = DataLoader(dm, shuffle=True, batch_size=1)

    dataTrain = tqdm(dataTrain)
    allloss = 0
    max_xlk=0
    flsall = 0
    lsall = 0
    psall=0
    for index,(gj) in enumerate(dataTrain):
        outx, outy, j = model(gj)
        loss, fls, ls, ps = criterion(outx, outy, gj, j)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        allloss += loss.item()
        flsall += fls.item()
        lsall += ls.item()
        psall += ps.item()

        dataTrain.set_description(desc="epoch {}, loss {}, fanloss {}, lsloss {}, psloss {}".format(epoch, allloss/(index+1), flsall/(index+1),lsall/(index+1),psall/(index+1)))
        max_xlk = allloss/(index+1)

    if max_xl > max_xlk:
        torch.save(model, './modelyzm.pth')
        max_xl = max_xlk



