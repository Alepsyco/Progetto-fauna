import pandas as pd
import numpy as np
import torch
from pandas import notna
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv, GCNConv
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import copy
import random

#Carica dataset finali di R

nodi = pd.read_csv("datasetNODES_V2.csv")  #habitat per comune + meteo per comune + incidenti per comune con target di previsione
adj = pd.read_csv("matriceADJ.csv", index_col=0)    #matrix adiacenza comuni
comuni = pd.read_csv("comuni.csv") #dati comuni (nome, perimetro, area, poligono 2d)


torch.manual_seed(1984)
np.random.seed(1984)
random.seed(1984)
#------------------------------------------------
#PROBLEMI CON ALLINAMENTO NOME ID, risolvo direttamente su py

orderedNODES = (nodi[["NOME", "ID"]].drop_duplicates().sort_values("ID"))
nomiNODES = orderedNODES["NOME"].tolist()
adj = adj.loc[nomiNODES, nomiNODES]

#costruzione archi index dalla matrice di adiacenza

adjBOOL = adj.values.astype(bool) #conversione matrice in bool
righeTRUE, colonneTRUE = np.where(adjBOOL)  #valori TRUE

archiIDX = torch.tensor(np.array([righeTRUE, colonneTRUE]), dtype=torch.long) #crea struttura dati pytorch per calcoli NN

#PROVA media incidenti dei mesi più vicini come feature
nodi = nodi.sort_values(["NOME", "mese"])
nodi["n_incidenti_PREV2"] = nodi.groupby("NOME")["n_incidenti_PREV"].transform(lambda x: x.rolling(2, min_periods=1).mean())
nodi["n_incidenti_PREV4"] = nodi.groupby("NOME")["n_incidenti_PREV"].transform(lambda x: x.rolling(4, min_periods=1).mean())

#I dati sono insufficienti, ho bisogno di usare più mesi indietro senza compromettere stagionalità
#ES aumento lookback, vado su mesi di riproduzione, ora siamo in mese di "letargo" ma viene rilevato pericolo

#trasforma numero mese in "cerchio" permette quindi di mostrare stagionalità (1->12->1)
nodi["sin_mese"] = np.sin(2 * np.pi * nodi["monthnum"] / 12)
nodi["cos_mese"] = np.cos(2 * np.pi * nodi["monthnum"] / 12)

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
    "sin_mese",
    "cos_mese"
]

#------------------------------------------------
#IDEA USARE GCNN + GATED RECURRENT UNIT (tipo di NN che analizza pattern di sequenze temporali)

#crea elenco dei mesi in ordine per GRU
monthORDERED = sorted(nodi["mese"].unique())

grafiMONTH = []  #lista che conterrà un oggetto Data per ciascun mese

#ordina per ID
for m in monthORDERED:
    nodiMESE = (nodi[nodi["mese"] == m].set_index("NOME").loc[nomiNODES].reset_index()) #usa nome per confrontare al posto di ID

    x = torch.tensor(nodiMESE[feature].fillna(0).values, dtype=torch.float)    #per feature
    y = torch.tensor(nodiMESE["target"].values, dtype=torch.float)        #per target previsione

#crea grafo mensile
    grafo = Data(x=x, edge_index=archiIDX, y=y)
    grafiMONTH.append(grafo)




xMONTH = torch.stack([g.x for g in grafiMONTH])    #feature ogni mese 255 comuni con X feature ciascuno
yMONTH = torch.stack([g.y for g in grafiMONTH])    #target (per misurare correttezza training)


#-----------------------------------------------------
#-----------------------------------------------------

#Architettura NN

class RischioGNN(nn.Module):
    def __init__(self, inputfeature, nodiGCNN, nodiGRU):
        super().__init__()

        # forma NN (ogni riga ha input e output)
        self.node_emb = nn.Sequential(
            nn.Linear(inputfeature, nodiGCNN),  # gcnn aggrega nodi vicini
            nn.ReLU())
        self.gcn = GATConv(inputfeature, nodiGCNN, heads=2, concat=False)
        self.gcn2 = GATConv(nodiGCNN, nodiGCNN, heads=2, concat=False)

        self.norm = nn.LayerNorm(nodiGCNN)
        self.dropout = nn.Dropout(0.3)
        self.gru = nn.GRU(input_size=nodiGCNN + inputfeature, hidden_size=nodiGRU, batch_first=True, bidirectional=False) #gru per temporale
        input_classifier = nodiGRU + nodiGRU + inputfeature
        self.output = nn.Sequential(
            nn.Linear(input_classifier, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

#funzione NN
    def forward(self, xSEQ, edgeIDX):
        numMONTH, numNODES, _ = xSEQ.shape #xSEQ ha shape [numero mesi, numero nodi (215?), numero feature]

        outputGCNN = []  #conterrà l'output della GCN per ogni mese

        for t in range(numMONTH): #per ogni mese t GCN aggrega feature dei nodi con vicini (con edge index)
            gcn_out = self.gcn(xSEQ[t], edgeIDX)
            local_out = self.node_emb(xSEQ[t])
            templayer = torch.relu(self.norm(gcn_out + local_out))
            gcn_out2 = self.gcn2(templayer, edgeIDX)        #aggiunto gcn secondo layer per vedere vicini dei vicini

            valoriMONTH = self.norm(gcn_out2 + templayer) #res connection (gcn + feature)
            valoriMONTH = torch.relu(valoriMONTH)
            valoriMONTH = self.dropout(valoriMONTH) #spegne casualmente neuroni per limitare overfitting

            valoriMONTH = torch.cat([valoriMONTH, xSEQ[t]], dim=-1)
            outputGCNN.append(valoriMONTH)

        outputGCNN = torch.stack(outputGCNN) #da 2D [numNODES, nodiGCNN] a 3D [numMONTH, numNODES, nodiGCNN] (per ogni mese tutti i comuni in ordine)
        outputGCNN = outputGCNN.permute(1, 0, 2) #riordina le dimensioni per GRU -> per ogni comune tutti i mesi in ordine

        outputGRU_month, _ = self.gru(outputGCNN) #GRU analizza andamento ogni mese (temp è singolo mese, final è aggregazione su comune)

        last_gru = outputGRU_month[:, -1, :]    #stato della rete all'ultimo mese
        mean_gru = outputGRU_month.mean(dim=1)  #media degli stati su finestra temporale
        raw_last = xSEQ[-1]     #feature originali, per res connection

        rischio = self.output(torch.cat([last_gru, mean_gru, raw_last], dim=1))
        return rischio.squeeze()

        #rischio = self.output(outputGRU_month[:, -1, :]) #[:, -1, :] -> per tutti i comuni prendi solo l'ultimo mese con tutti i nodiGRU))
        #return rischio.squeeze() #squeeze rimuove dimensioni superflue


windowlenght = 6  #mesi di lookback

xSEQ= []
ySEQ = []

for t in range(len(monthORDERED) - windowlenght):    #per ogni blocco da windowlenght
    windowX = xMONTH[t: t + windowlenght]          #definisce input
    targetY = yMONTH[t + windowlenght]             #definisce month da prevedere
    xSEQ.append(windowX)
    ySEQ.append(targetY)


trainQUOTA = 0.75
valdationQUOTA = 0.10


splitTRAIN = int(len(xSEQ) * trainQUOTA)
splitVAL = int(len(xSEQ) * (trainQUOTA + valdationQUOTA))

xTRAIN = xSEQ[:splitTRAIN]
yTRAIN = ySEQ[:splitTRAIN]
xVAL = xSEQ[splitTRAIN:splitVAL]
yVAL = ySEQ[splitTRAIN:splitVAL]
xTEST = xSEQ[splitVAL:]
yTEST = ySEQ[splitVAL:]


#normalizzazione (NON TUTTI I VALORI VANNO NORMALIZZATI, NO LUNA NO SIN COS)
notnorm = [c for c in feature if c.startswith('sin_') or c.startswith('cos_')]      #se si aggiungono feature da non normalizzare metti qui
idx_norm = [i for i, f in enumerate(feature) if f not in notnorm]    #numera feature da normalizzare

#media e dev standard
mediaTRAIN = torch.cat([t.reshape(-1, len(feature))[:, idx_norm] for t in xTRAIN]).mean(dim=0)
devstandardTRAIN = torch.cat([t.reshape(-1, len(feature))[:, idx_norm] for t in xTRAIN]).std(dim=0) + 1e-9

#per ogni elemento da standardizzare
#CLONE FONDAMENTALE PER NON MODIFICARE I VALORI ORIGINALI
def norm(lista_seq):
    normalized = []
    for t in lista_seq:
        t_clone = t.clone()
        t_clone[:, :, idx_norm] = (t_clone[:, :, idx_norm] - mediaTRAIN) / devstandardTRAIN
        normalized.append(t_clone)
    return normalized

#normalizzazione dei vari set
xTRAIN = norm(xTRAIN)
xVAL = norm(xVAL)
xTEST = norm(xTEST)

pos = sum(t.sum() for t in yTRAIN)          #totale target=1 nel training
neg = sum(t.numel() for t in yTRAIN) - pos    #totale target=0 nel training

#TODO SCEGLI TRA RADICE O RAPPORTO (NOTA LEGACY?)
posWG = torch.tensor([0.8 * torch.sqrt(neg / pos)], dtype=torch.float) #quanto pesare di più gli esempi positivi nella loss (VEDI SOTTO train/test)
#posWG = torch.tensor([neg / pos], dtype=torch.float)

####################################################################################################################


#parametri

parametri = RischioGNN(
    inputfeature=len(feature),
    nodiGCNN=32,
    nodiGRU=64

)

epochs = 300
learningrate = 0.002
weightdecay = 0.0008

stepsize = 35 #ogni x epoche
gammaLR = 0.7   #moltiplica per y (diminuisce)

#per early stopping
patience = 100  #n epoche di attesa prima di fermarsi (evita che si possano ignorare possibilim miglioramenti





####################################################################################################################


#train

patienceCOUNT = 0
aucBEST = 0.0
bestweights = None

lossFN = nn.BCEWithLogitsLoss(pos_weight=posWG)

optimz = optim.Adam(parametri.parameters(), lr=learningrate, weight_decay=weightdecay) #Adam aggiorna i pesi della rete per minimizzare la loss, lr = velocità di apprendimento

#scheduler per modificare LR durante training
scheduler = optim.lr_scheduler.StepLR(optimz, step_size=stepsize, gamma=gammaLR) #ogni TOT epoche LR*gamma (diminuisce)


for epoca in range(epochs):
    parametri.train()
    lossEPOCA = 0

    for i in range(len(xTRAIN)):
        optimz.zero_grad()      #azzera i gradienti calcolati nell'epoca precedente

        previsioneTRAIN = parametri(xTRAIN[i], archiIDX) #passa la sequenza di training nel modello e ottieni la previsione su mese (come da architettura)
        target = yTRAIN[i] #confronto target con valore effettivo dell'ultimo mese

        # TROPPI VALORI A 0, AGGIUSTO LOSS DI MODO CHE PREVISIONE SBAGLIATA SU 1 SIA MOLTO DISINCENTIVATA

        lossTRAIN = lossFN(previsioneTRAIN, target)

        lossTRAIN.backward()                 #calcola gradienti (quanto ogni peso ha contribuito a errore)
        optimz.step() #aggiorna pesi della rete in base a gradienti

        lossEPOCA += lossTRAIN.item() #aggiornamento loss totale

    scheduler.step() #eventuale modifica LR vedi sopra


    #EARLY STOPPING
    parametri.eval()
    previsioniVAL_temp = []
    targetVAL_temp = []
    with torch.no_grad():
        for i in range(len(xVAL)):
            previsioniVAL_temp.append(torch.sigmoid(parametri(xVAL[i], archiIDX)).detach().cpu()) #detach cpu per non intasare
            targetVAL_temp.append(yVAL[i].cpu())

    aucEPOCA = roc_auc_score(torch.cat(targetVAL_temp).numpy(), torch.cat(previsioniVAL_temp).numpy()) #calcola aucroc per epoch

    if aucEPOCA > aucBEST:
        aucBEST = aucEPOCA
        patienceCOUNT = 0
        bestweights = copy.deepcopy(parametri.state_dict())
    else:
        patienceCOUNT += 1

    if epoca % 30 == 0:
        print(f"Epoch: {epoca} || Loss media training: {lossEPOCA / len(xTRAIN):.4f}")

    if patienceCOUNT >= patience:
        print(f"\n##################################################")
        print(f"Early stopping attivato all'epoca {epoca}")
        print(f"##################################################\n")
        break

#ricarica i pesi migliori ottenuti prima del blocco/overfitting
if bestweights is not None:
    parametri.load_state_dict(bestweights)


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

        #converto i logits (valori grezzi di uscita) in probabilita via sigmoid per la valutazione
        previsioneVAL_prob = torch.sigmoid(previsioneVAL)

        lossVAL += lossFN(previsioneVAL, target).item()

        previsioniVAL_idx.append(previsioneVAL_prob)
        targetVAL_idx.append(target)

print("Loss media validation:", lossVAL / len(xVAL))

#concatenazione dei tensor serve per calcolo metriche sotto
yTARGET = torch.cat(targetVAL_idx).numpy()
yFORECAST = torch.cat(previsioniVAL_idx).numpy()

#la soglia di 0,5 messa a caso non andava bene, la trovo usando f1 score su validation set
precisions, recalls, thresholds = precision_recall_curve(yTARGET, yFORECAST)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10) #media armonica di precision e recall

#matrice TRUE FALSE con soglia mobile (CALCOLATA TRAMITE F1SCORE)
truePOS = ((yFORECAST > thresholds[np.argmax(f1_scores)]) & (yTARGET == 1)).sum()
falsePOS = ((yFORECAST > thresholds[np.argmax(f1_scores)]) & (yTARGET == 0)).sum()
falseNEG = ((yFORECAST <= thresholds[np.argmax(f1_scores)]) & (yTARGET == 1)).sum()
trueNEG = ((yFORECAST <= thresholds[np.argmax(f1_scores)]) & (yTARGET == 0)).sum()

print("Soglia usata (in precedenza 0,5):", thresholds[np.argmax(f1_scores)])

print("\nTrue positive:", truePOS)
print("False positive:", falsePOS)
print("True negative:", trueNEG)
print("False negative:", falseNEG)


recallPOS = truePOS / (truePOS + falseNEG + 1e-10) #recall (IMPORTANTE, MEGLIO FALSI POSITIVI CHE FALSI NEGATIVI)
precisionPOS = truePOS / (truePOS + falsePOS + 1e-10) #precision

print("\nRecall positive:", recallPOS)
print("Precision positive:", precisionPOS)

aucroc = roc_auc_score(yTARGET, yFORECAST) #quanto il modello distingue categorie 1/0 (intorno a 50% male)
averageprecision = average_precision_score(yTARGET, yFORECAST)

print("AUC-ROC:", aucroc)
print("Average Precision:", averageprecision)

################

#TODO SOSTITUISCI lossFN con focal loss

