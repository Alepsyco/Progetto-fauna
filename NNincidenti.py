import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.optim as optim

#Carica dataset finali di R

nodi = pd.read_csv("datasetNODES.csv")  #habitat per comune + meteo per comune + incidenti per comune con target di previsione
adj = pd.read_csv("matriceADJ.csv", index_col=0)    #matrix adiacenza comuni
comuni = pd.read_csv("comuni.csv") #dati comuni (nome, perimetro, area, poligono 2d)

#------------------------------------------------
#PROBLEMI CON ALLINAMENTO NOME ID, risolvo direttamente su py

orderedNODES = (nodi[["NOME", "ID"]].drop_duplicates().sort_values("ID"))
nomiNODES = orderedNODES["NOME"].tolist()
adj = adj.loc[nomiNODES, nomiNODES]

#costruzione archi index dalla matrice di adiacenza

adjBOOL = adj.values.astype(bool) #conversione matrice in bool
righeTRUE, colonneTRUE = np.where(adjBOOL)  #valori TRUE

archiIDX = torch.tensor(np.array([righeTRUE, colonneTRUE]), dtype=torch.long) #crea struttura dati pytorch per calcoli NN


#------------------------------------------------
#FEATURE PER NODO E (TODO: AGGIUNGI FEATURE STRADE E ALTEZZE CIRCOSTANTI)

feature = [
    "sensecoM",
    "pressantroM",
    "valecoM",
    "tempMEAN",
    "precipMEAN",
    "ventoMEAN",
    "n_incidenti"
]

#------------------------------------------------
#IDEA USARE GCNN + GATED RECURRENT UNIT (tipo di NN che analizza pattern di sequenze temporali)

#crea elenco delle settimane in ordine per GRU
weekORDERED = sorted(nodi["settimana"].unique())

grafiWEEK = []  #lista che conterrà un oggetto Data per ciascuna settimana

#ordina per ID
for sett in weekORDERED:
    nodiSETTIMANA = (nodi[nodi["settimana"] == sett].set_index("NOME").loc[nomiNODES].reset_index()) #usa nome per confrontare al posto di ID

    x = torch.tensor(nodiSETTIMANA[feature].fillna(0).values, dtype=torch.float)    #per feature
    y = torch.tensor(nodiSETTIMANA["target"].values, dtype=torch.float)        #per target previsione

#crea grafo settimanale
    grafo = Data(x=x, edge_index=archiIDX, y=y)
    grafiWEEK.append(grafo)




xWEEK = torch.stack([g.x for g in grafiWEEK])    #feature ogni settimana 255 comuni con X feature ciascuno
yWEEK = torch.stack([g.y for g in grafiWEEK])    #target (per misurare correttezza training)


#-----------------------------------------------------
#-----------------------------------------------------

#Architettura NN

class RischioGNN(nn.Module):
    def __init__(self, inputfeature, nodiGCNN, nodiGRU):
        super().__init__()

        #forma NN (ogni riga ha input e output)
        self.gcn = GCNConv(inputfeature, nodiGCNN) #gcnn aggregra feature dei vicini a nodo
        self.gru = nn.GRU(input_size=nodiGCNN, hidden_size=nodiGRU, batch_first=True) #gru per temporale
        self.output = nn.Linear(nodiGRU, 1)
        self.sigmoid = nn.Sigmoid()

#funzione NN
    def forward(self, xSEQ, edgeIDX):
        numWEEK, numNODES, _ = xSEQ.shape #xSEQ ha shape [numero settimane, numero nodi (215?), numero feature]

        outputGCNN = []  #conterrà l'output della GCN per ogni settimana

        for t in range(numWEEK): #per ogni settimana t GCN aggrega feature dei nodi con vicini (con edge index)
            valoriWEEK = self.gcn(xSEQ[t], edgeIDX)
            valoriWEEK = torch.relu(valoriWEEK)  #RELU TODO VERIFICA
            outputGCNN.append(valoriWEEK)

        outputGCNN = torch.stack(outputGCNN) #da 2D [numNODES, nodiGCNN] a 3D [numWEEK, numNODES, nodiGCNN] (per ogni settimana tutti i comuni in ordine)
        outputGCNN = outputGCNN.permute(1, 0, 2) #riordina le dimensioni per GRU -> per ogni comune tutte le settimane in ordine

        outputGRU_week, outputGRU_final = self.gru(outputGCNN) #GRU analizza andamento ogni settimana (temp è singola settimana, final è aggregazione su comune)

        rischio = self.sigmoid(self.output(outputGRU_week[:, -1, :])) #[:, -1, :] -> per tutti i comuni prendi solo l'ultima settimana con tutti i nodiGRU))
        return rischio.squeeze() #squeeze rimuove dimensioni superflue


windowlenght = 8  #settimane di lookback

xSEQ= []
ySEQ = []

for t in range(len(weekORDERED) - windowlenght):    #per ogni blocco da windowlenght
    windowX = xWEEK[t: t + windowlenght]          #definisce input
    targetY = yWEEK[t + windowlenght]             #definisce week da prevedere
    xSEQ.append(windowX)
    ySEQ.append(targetY)




#divido train e test con %
traintest = 0.8                                                            #MODIFICARE PER TRAIN/TEST

split = int(len(xSEQ) * traintest)

xTRAIN = xSEQ[:split]   #da prima fino a split
yTRAIN = ySEQ[:split]
xTEST = xSEQ[split:]    #da dopo split
yTEST = ySEQ[split:]

#ADDED normalizzazione perchè training troppo lento
mediaTRAIN = torch.cat([t.reshape(-1, len(feature)) for t in xTRAIN]).mean(dim=0) #appiattisce per mantenere solo feature e fa media
devstandardTRAIN = torch.cat([t.reshape(-1, len(feature)) for t in xTRAIN]).std(dim=0) + 1e-9 #ditto deviazione standard

#normalizzazione di tutti i valori
xTRAIN = [(t - mediaTRAIN) / devstandardTRAIN for t in xTRAIN]
xTEST = [(t - mediaTRAIN) / devstandardTRAIN for t in xTEST]


pos = sum(t.sum() for t in yTRAIN)          #totale target=1 nel training
neg = sum(t.numel() for t in yTRAIN) - pos    #totale target=0 nel training

posWG = neg / pos * 0.6  #quanto pesare di più gli esempi positivi nella loss (VEDI SOTTO train/test)


####################################################################################################################


#parametri TODO vari TEST PER OVER/UNDERFITTING

parametri = RischioGNN(
    inputfeature=7,
    nodiGCNN=32,
    nodiGRU=32

)
epochs = 300
learningrate = 0.01
weightdecay = 0

stepsize = 60
gammaLR = 0.5


####################################################################################################################






#train e test

lossFN = nn.BCELoss(weight=None)  #BCE binary cross entropy per binario
optimz = optim.Adam(parametri.parameters(), lr=learningrate, weight_decay=weightdecay) #Adam aggiorna i pesi della rete per minimizzare la loss, lr = velocità di apprendimento

#ADDED scheduler per modificare LR durante training
scheduler = optim.lr_scheduler.StepLR(optimz, step_size=stepsize, gamma=gammaLR) #ogni TOT epoche LR*gamma (diminuisce)


for epoca in range(epochs):
    parametri.train()
    lossEPOCA = 0

    for i in range(len(xTRAIN)):
        optimz.zero_grad()      #azzera i gradienti calcolati nell'epoca precedente

        previsioneTRAIN = parametri(xTRAIN[i], archiIDX) #passa la sequenza di training nel modello e ottieni la previsione su settimana (come da architettura)
        target = yTRAIN[i] #confronto target con valore effettivo dell'ultima settimana

        # TROPPI VALORI A 0, AGGIUSTO LOSS DI MODO CHE PREVISIONE SBAGLIATA SU 1 SIA MOLTO DISINCENTIVATA
        pesi = torch.where(target == 1, posWG, torch.tensor(1.0, dtype=torch.float))
        lossTRAIN = nn.functional.binary_cross_entropy(previsioneTRAIN, target, reduction='none') #calcolo loss per ogni comune
        lossTOT_train = (lossTRAIN * pesi).mean()

        lossTOT_train.backward() #calcola gradienti (quanto ogni peso ha contribuito a errore)
        optimz.step() #aggiorna pesi della rete in base a gradienti

        lossEPOCA += lossTOT_train.item() #aggiornamento loss totale

    scheduler.step() #eventuale modifica LR vedi sopra

    if epoca % 20 == 0:
        print(f"Epoch: {epoca} Loss media training: {lossEPOCA / len(xTRAIN):.4f}")


parametri.eval()
with torch.no_grad():   #no backprop in test set
    lossTEST = 0
    accuratezzaTOT = 0

    for i in range(len(xTEST)):
        previsioneTEST = parametri(xTEST[i], archiIDX)
        target = yTEST[i]

        lossTEST += nn.functional.binary_cross_entropy(previsioneTEST, target).item()
        accuratezzaTOT += ((previsioneTEST > 0.5).float() == target).float().mean().item()

    print("Loss media test:", lossTEST / len(xTEST))
    print("Accuratezza media test:", accuratezzaTOT / len(xTEST))


################

#TODO AGGIUNGI FASI LUNARI (PROBABILMENTE MEGLIO FARE 4 COLONNE CON ONESHOTVECTOR?)