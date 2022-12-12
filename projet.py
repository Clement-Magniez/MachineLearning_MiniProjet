import torch.nn.functional as F
import torch.nn as nn
import torch
import sys

carrosserie = 24
carburant = 6
marque = 0
conso = 11

marque_dict = {}
carburant_dict = {}
carroserie_dict = {}

x_carrosserie = []
x_carburant = []
x_conso = []

y_marque = []

class Mod(nn.Module):
    def __init__(self,sizes):
        super(Mod, self).__init__()
        self.flatten = nn.Flatten()
        stack = [nn.Linear(sizes[i//2], sizes[i//2+1]) if i%2==0 \
        else nn.ReLU() for i in range(2*len(sizes)-3)]
        self.stack = nn.Sequential(*stack)

    def forward(self,x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits

def test(mod, testloader, crit):
    mod.train(False)
    totloss, nbatch = 0., 0
    for data in testloader:
        inputs, goldy = data
        haty = mod(inputs)
        loss = crit(haty,goldy)
        totloss += loss.item()
        nbatch += 1
    totloss /= float(nbatch)
    mod.train(True)
    return totloss

def train(mod, trainloader, testloader, crit, nepochs=20):
    optim = torch.optim.Adam(mod.parameters(), lr=0.001)
    for epoch in range(nepochs):
        print(epoch)
        testloss = test(mod, testloader, crit)
        totloss, nbatch = 0., 0
        for data in trainloader:
            inputs, goldy = data
            optim.zero_grad()
            haty = mod(inputs)
            loss = crit(haty,goldy)
            totloss += loss.item()
            nbatch += 1
            loss.backward()
            optim.step()
        totloss /= float(nbatch)
        print("err",totloss,testloss)
    print("fin",totloss,testloss,file=sys.stderr)


def load_data():
    temp_carrosserie = {}
    with open("mars-2014-complete.csv","r") as f: ls=f.readlines()
    for l in ls[1:]:
        d = l.split(';')

        if '' in (d[conso], d[carrosserie], d[marque], d[carburant]): continue
        if d[carrosserie] == 'MINIBUS': continue

        d_carrosserie = d[carrosserie]
        if d_carrosserie not in carroserie_dict: carroserie_dict[d_carrosserie] = len(carroserie_dict)
        if d_carrosserie not in temp_carrosserie: temp_carrosserie[d_carrosserie] = 0
        temp_carrosserie[d_carrosserie] += 1
        d_carburant = d[carburant]
        if d_carburant not in carburant_dict: carburant_dict[d_carburant] = len(carburant_dict)
        d_marque = d[marque]
        if d_marque not in marque_dict: marque_dict[d_marque] = len(marque_dict)

        d_conso = float(d[conso].replace(',', '.'))

        x_carrosserie.append(carroserie_dict[d_carrosserie])
        x_carburant.append(carburant_dict[d_carburant])
        x_conso.append(d_conso)
        y_marque.append(marque_dict[d_marque])


    print(temp_carrosserie)

    t_carrosserie = F.one_hot(torch.LongTensor(x_carrosserie), len(carroserie_dict)).float()
    t_carburant = F.one_hot(torch.LongTensor(x_carburant), len(carburant_dict)).float()
    t_conso = torch.FloatTensor(x_conso)
    datax = torch.cat((t_carrosserie, t_carburant, t_conso.reshape(t_conso.shape[0], 1)), dim=1)
    print(datax.shape)

    t_marque = F.one_hot(torch.LongTensor(y_marque), len(marque_dict)).float()
    datay = torch.Tensor(t_marque)
    print(datay.shape)

    r = torch.randperm(datax.shape[0])
    datax, datay = datax[r], datay[r]

    f = int(datax.shape[0] / 8)
    testx, trainx = datax[:f], datax[f:]
    testy, trainy = datay[:f], datay[f:]



    trainds = torch.utils.data.TensorDataset(trainx, trainy)
    trainloader = torch.utils.data.DataLoader(trainds, batch_size=1, shuffle=True)
    testds = torch.utils.data.TensorDataset(testx, testy)
    testloader = torch.utils.data.DataLoader(testds, batch_size=1, shuffle=True)
    return trainloader, testloader


trainloader, testloader = load_data()
crit = nn.MSELoss()
mod = Mod([22, 10, 44])
print("nparms",sum(p.numel() for p in mod.parameters() if p.requires_grad),file=sys.stderr)
train(mod, trainloader, testloader, crit)