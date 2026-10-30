import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score

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

nodi = pd.get_dummies(nodi, columns=["fase_lunare"], prefix="luna", dtype=float)    #prende elementi quantitativi fa one hot vector con prefisso luna_ (ex luna_full"

#PROVA media incidenti delle settimane più vicine come feature
nodi = nodi.sort_values(["NOME", "settimana"])
nodi["n_incidenti_PREV2"] = nodi.groupby("NOME")["n_incidenti_PREV"].transform(lambda x: x.rolling(2, min_periods=1).mean())
nodi["n_incidenti_PREV4"] = nodi.groupby("NOME")["n_incidenti_PREV"].transform(lambda x: x.rolling(4, min_periods=1).mean())


#------------------------------------------------
#FEATURE PER NODO E (TODO: AGGIUNGI FEATURE STRADE E ALTEZZE CIRCOSTANTI)

feature = [
    "sensecoM",
    "pressantroM",
    "valecoM",
    "tempMEAN",
    "precipMEAN",
    "ventoMEAN",
    "n_incidenti_PREV",
    "n_incidenti_PREV2",
    "n_incidenti_PREV4",
    "luna_New",
    "luna_Waxing",
    "luna_Full",
    "luna_Waning"
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
        self.gcn = SAGEConv(inputfeature, nodiGCNN) #gcnn aggregra feature dei vicini a nodo
        self.norm = nn.LayerNorm(nodiGCNN)
        self.dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(input_size=nodiGCNN, hidden_size=nodiGRU, batch_first=True) #gru per temporale
        self.output = nn.Linear(nodiGRU, 1)  #NO SIGMOID uso BCElogit

#funzione NN
    def forward(self, xSEQ, edgeIDX):
        numWEEK, numNODES, _ = xSEQ.shape #xSEQ ha shape [numero settimane, numero nodi (215?), numero feature]

        outputGCNN = []  #conterrà l'output della GCN per ogni settimana

        for t in range(numWEEK): #per ogni settimana t GCN aggrega feature dei nodi con vicini (con edge index)
            valoriWEEK = self.gcn(xSEQ[t], edgeIDX)
            valoriWEEK = self.norm(valoriWEEK)
            valoriWEEK = torch.relu(valoriWEEK)  #RELU TODO VERIFICA
            valoriWEEK = self.dropout(valoriWEEK)
            outputGCNN.append(valoriWEEK)

        outputGCNN = torch.stack(outputGCNN) #da 2D [numNODES, nodiGCNN] a 3D [numWEEK, numNODES, nodiGCNN] (per ogni settimana tutti i comuni in ordine)
        outputGCNN = outputGCNN.permute(1, 0, 2) #riordina le dimensioni per GRU -> per ogni comune tutte le settimane in ordine

        outputGRU_week, outputGRU_final = self.gru(outputGCNN) #GRU analizza andamento ogni settimana (temp è singola settimana, final è aggregazione su comune)

        rischio = self.output(outputGRU_week[:, -1, :]) #[:, -1, :] -> per tutti i comuni prendi solo l'ultima settimana con tutti i nodiGRU))
        return rischio.squeeze() #squeeze rimuove dimensioni superflue


windowlenght = 8  #settimane di lookback

xSEQ= []
ySEQ = []

for t in range(len(weekORDERED) - windowlenght):    #per ogni blocco da windowlenght
    windowX = xWEEK[t: t + windowlenght]          #definisce input
    targetY = yWEEK[t + windowlenght]             #definisce week da prevedere
    xSEQ.append(windowX)
    ySEQ.append(targetY)





trainQUOTA = 0.7
valdationQUOTA = 0.15


splitTRAIN = int(len(xSEQ) * trainQUOTA)
splitVAL = int(len(xSEQ) * (trainQUOTA + valdationQUOTA))

xTRAIN = xSEQ[:splitTRAIN]
yTRAIN = ySEQ[:splitTRAIN]
xVAL = xSEQ[splitTRAIN:splitVAL]
yVAL = ySEQ[splitTRAIN:splitVAL]
xTEST = xSEQ[splitVAL:]
yTEST = ySEQ[splitVAL:]


#ADDED normalizzazione perchè training troppo lento
mediaTRAIN = torch.cat([t.reshape(-1, len(feature)) for t in xTRAIN]).mean(dim=0) #appiattisce per mantenere solo feature e fa media
devstandardTRAIN = torch.cat([t.reshape(-1, len(feature)) for t in xTRAIN]).std(dim=0) + 1e-9 #ditto deviazione standard

#normalizzazione di tutti i valori
xTRAIN = [(t - mediaTRAIN) / devstandardTRAIN for t in xTRAIN]
xVAL= [(t - mediaTRAIN) / devstandardTRAIN for t in xVAL]
xTEST = [(t - mediaTRAIN) / devstandardTRAIN for t in xTEST]


pos = sum(t.sum() for t in yTRAIN)          #totale target=1 nel training
neg = sum(t.numel() for t in yTRAIN) - pos    #totale target=0 nel training

posWG = torch.tensor([1,25*np.sqrt(float(neg) / float(pos))],dtype=torch.float) #quanto pesare di più gli esempi positivi nella loss (VEDI SOTTO train/test)


####################################################################################################################


#parametri TODO vari TEST PER OVER/UNDERFITTING

parametri = RischioGNN(
    inputfeature=13,
    nodiGCNN=32,
    nodiGRU=32

)
epochs = 300
learningrate = 0.002
weightdecay = 0.0005

stepsize = 100
gammaLR = 0.5


####################################################################################################################






#train e test

lossFN = nn.BCEWithLogitsLoss(pos_weight=posWG)  #BCE binary cross entropy per binario TODO LOGIT
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

        lossTRAIN = lossFN(previsioneTRAIN, target)

        lossTRAIN.backward()                 #calcola gradienti (quanto ogni peso ha contribuito a errore)
        optimz.step() #aggiorna pesi della rete in base a gradienti

        lossEPOCA += lossTRAIN.item() #aggiornamento loss totale

    scheduler.step() #eventuale modifica LR vedi sopra

    if epoca % 60 == 0:
        print(f"Epoch: {epoca} || Loss media training: {lossEPOCA / len(xTRAIN):.4f}")


#-----------------------------------------------------
#-----------------------------------------------------
#VALIDATION

truePOS = 0
falsePOS = 0
trueNEG = 0
falseNEG = 0

parametri.eval()

previsioniVAL_idx = []
targetVAL_idx = []

with torch.no_grad():
    lossVAL = 0
    accuratezzaVAL = 0

    for i in range(len(xVAL)):
        previsioneVAL = parametri(xVAL[i], archiIDX)
        target = yVAL[i]

        # Convertiamo i logits (valori grezzi di uscita) in probabilita via sigmoid per la valutazione
        previsioneVAL_prob = torch.sigmoid(previsioneVAL)

        lossVAL += lossFN(previsioneVAL, target).item()
        accuratezzaVAL += ((previsioneVAL_prob > 0.5) == target).float().mean().item()

        previsioniVAL_idx.append(previsioneVAL_prob)
        targetVAL_idx.append(target)

print("Loss media validation:", lossVAL / len(xVAL))
print("Accuratezza media validation:", accuratezzaVAL / len(xVAL))

# Concatenazione dei tensor per calcolo metriche scikit-learn
yTARGET = torch.cat(targetVAL_idx).numpy()
yFORECAST = torch.cat(previsioniVAL_idx).numpy()

# EDIT la soglia di 0,5 non andava bene, la trovo usando f1 score su validation set
precisions, recalls, thresholds = precision_recall_curve(yTARGET, yFORECAST)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10) #media armonica di precisione e recupero

#matrice TRUE FALSE con soglia mobile (CALCOLATA TRAMITE F1SCORE)
truePOS = ((yFORECAST > thresholds[np.argmax(f1_scores)]) & (yTARGET == 1)).sum()
falsePOS = ((yFORECAST > thresholds[np.argmax(f1_scores)]) & (yTARGET == 0)).sum()
falseNEG = ((yFORECAST <= thresholds[np.argmax(f1_scores)]) & (yTARGET == 1)).sum()
trueNEG = ((yFORECAST <= thresholds[np.argmax(f1_scores)]) & (yTARGET == 0)).sum()

print("Soglia usata (in precedenza 0,5):", thresholds[np.argmax(f1_scores)])
print("True positive:", truePOS)
print("False positive:", falsePOS)
print("True negative:", trueNEG)
print("False negative:", falseNEG)


recallPOS = truePOS / (truePOS + falseNEG + 1e-10) #recall
precisionPOS = truePOS / (truePOS + falsePOS + 1e-10) #precision

print("Recall positive:", recallPOS)
print("Precision positive:", precisionPOS)

aucroc = roc_auc_score(yTARGET, yFORECAST) #quanto il modello distingue categorie 1/0 (intorno a 50% male)
averageprecision = average_precision_score(yTARGET, yFORECAST)

print("AUC-ROC:", aucroc)
print("Average Precision:", averageprecision)

################

#TODO PROVARE EARLY STOPPING?