import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
import copy
from scipy.spatial import cKDTree

######################################################################################################################################
# Analisi rischio strade (blocchi da 1km)
# Analisi incidenti mese per comune non molto significativo, leggermente meglio di baseline
# questo torna a idea originale (divisione in blocchi, distanza da bosco con 20 200 500 metri cerchio)
#
#
######################################################################################################################################

#Carica dataset finale da R

sezioni = pd.read_csv("sezioni.csv", index_col=0)     #un blocco per riga, l'indice è il numero di riga del blocco (da R, parte da 1)
incidenti = pd.read_csv("incidentiBLOCCHI.csv")       #un incidente per riga con il numero del blocco a cui appartiene

torch.manual_seed(1984)
np.random.seed(1984)

#------------------------------------------------
#divisione tempo e strada per training e creazione feature

trainQUOTA = 0.80
valdationQUOTA = 0.10     #il test è il resto
start = "2024-01-01"

specie = ["Cervo", "Cinghiale", "Daino", "Capriolo", "Lupo", "Camoscio", "Tasso", "Sciacallo dorato", "Volpe"]    #specie d'interesse
incidenti = incidenti[incidenti["specie"].isin(specie)]

#gli incidenti prima sono parte del training, dopo sono target
sezioni["incidentiBF"] = incidenti[incidenti["data_iso"] < start].groupby("blocco").size().reindex(sezioni.index, fill_value=0) #fill_value altrimenti da errore
sezioni["incidentiAF"] = incidenti[incidenti["data_iso"] >= start].groupby("blocco").size().reindex(sezioni.index, fill_value=0)

sezioni["target"] = (sezioni["incidentiAF"] > 0).astype(int)     #1 se il blocco avrà incidenti, 0 altrimenti

sezioni["altitudineDEVSTANDARD"] = sezioni["altitudineDEVSTANDARD"].fillna(0)     #NA se il blocco marca solo una cella raster (no devstandard)
strada = sezioni["TRIM_SUSTAINER"] + " " + sezioni["TRIM_STR_CODE_1"]     #sempre per distinguere SP1

stradarand = np.random.permutation(strada.unique()) #ordine random per strade

splitTRAIN = int(len(stradarand) * trainQUOTA)
splitVAL = int(len(stradarand) * (trainQUOTA + valdationQUOTA))

sezioniTRAIN = sezioni[strada.isin(stradarand[:splitTRAIN])]
sezioniVAL = sezioni[strada.isin(stradarand[splitTRAIN:splitVAL])]
sezioniTEST = sezioni[strada.isin(stradarand[splitVAL:])]

#------------------------------------------------
#FEATURE PER NODO E (TODO: altre feature)

feature = [
    "altitudineMEAN",
    "altitudineDEVSTANDARD",
    "bosco20",
    "bosco200",
    "bosco500",
    "edifici50",
    "edifici200",
    "lunghezza_m",
    "incidentiBF"
]



#normalizzazione (media e dev standard solo dal training)
mediaTRAIN = sezioniTRAIN[feature].mean()
devstandardTRAIN = sezioniTRAIN[feature].std() + 1e-9

####################################################################################################################

#parametri

epochs = 300
learningrate = 0.005
weightdecay = 0.005

#per early stopping
patience = 100  #n epoche di attesa prima di fermarsi

lossFN = nn.BCEWithLogitsLoss()

#-----------------------------------------------------

#NN
#creo rete grafo (nodi = blocchi) collegamento solo se sulla stessa strada continuativamente!!

posizione = dict(zip(zip(strada, sezioni["km"]), range(len(sezioni))))     #(strada, km) segna posizione del blocco nella tabella

righe = []     #blocco di partenza
colonne = []     #blocco di arrivo
for i, (via, km) in enumerate(zip(strada, sezioni["km"])):
    for vicino in [km - 1, km + 1]:
        if (via, vicino) in posizione:     #se esiste il blocco prima/dopo sulla stessa strada
            righe.append(i)
            colonne.append(posizione[(via, vicino)])

archiIDX = torch.tensor(np.array([righe, colonne]), dtype=torch.long)


class RischioGNN(nn.Module):
    def __init__(self, inputfeature, nodi1, nodi2):
        super().__init__()

        #forma NN (ogni riga ha input e output)
        self.node_emb = nn.Sequential(
            nn.Linear(inputfeature, nodi1),
            nn.ReLU())
        self.gcn = GATv2Conv(inputfeature, nodi1, heads=2, concat=False)     #aggrega feature blocchi vicini

        self.norm = nn.LayerNorm(nodi1)
        self.dropout = nn.Dropout(0.3)     #spegne casualmente neuroni per limitare overfitting

        self.output = nn.Sequential(
            nn.Linear(nodi1 + inputfeature, nodi2),
            nn.ReLU(),
            nn.Linear(nodi2, 1)
        )

#funzione NN
    def forward(self, xBLOCK, edgeIDX):
        gcn_out = self.gcn(xBLOCK, edgeIDX)
        local_out = self.node_emb(xBLOCK)

        valori = torch.relu(self.norm(gcn_out + local_out)) #res connection (vicini + feature del blocco)
        valori = self.dropout(valori)

        valori = torch.cat([valori, xBLOCK], dim=-1)

        rischio = self.output(valori)
        return rischio.squeeze()

#grafo di blocchi
xBLOCK = torch.tensor(((sezioni[feature] - mediaTRAIN) / devstandardTRAIN).values, dtype=torch.float)
yBLOCK = torch.tensor(sezioni["target"].values, dtype=torch.float)

idxTRAIN = np.where(strada.isin(stradarand[:splitTRAIN]).values)[0]     #posizioni dei blocchi nei due insiemi
idxVAL = np.where(strada.isin(stradarand[splitTRAIN:splitVAL]).values)[0]

#-----------------------------------------------------

#train

parametriGNN = RischioGNN(len(feature), 16, 8)

patienceCOUNT = 0
aucBEST = 0.0
bestweights = None

optimz = optim.Adam(parametriGNN.parameters(), lr=learningrate, weight_decay=weightdecay)


for epoca in range(epochs):
    parametriGNN.train()

    optimz.zero_grad() #azzera i gradienti calcolati nell'epoca precedente

    previsioneTRAIN = parametriGNN(xBLOCK, archiIDX) #passa tutto il grafo nel modello

    lossTRAIN = lossFN(previsioneTRAIN[idxTRAIN], yBLOCK[idxTRAIN]) #loss solo sui blocchi del training

    lossTRAIN.backward()  #calcola gradienti (quanto ogni peso ha contribuito a errore)
    optimz.step() #aggiorna pesi della rete in base a gradienti


    #EARLY STOPPING
    parametriGNN.eval()
    with torch.no_grad():
        previsioneVAL = torch.sigmoid(parametriGNN(xBLOCK, archiIDX))[idxVAL]

    aucEPOCA = roc_auc_score(yBLOCK[idxVAL].numpy(), previsioneVAL.numpy()) #calcola aucroc per epoca

    if aucEPOCA > aucBEST:
        aucBEST = aucEPOCA
        patienceCOUNT = 0
        bestweights = copy.deepcopy(parametriGNN.state_dict())
    else:
        patienceCOUNT += 1

    if epoca % 30 == 0:
        print(f"Epoch: {epoca} || Loss training: {lossTRAIN.item():.5f}")

    if patienceCOUNT >= patience:
        print("############################################")
        print("#                                          #")
        print(f"#   Early stopping attivato all'epoca {epoca}   #")
        print("#                                          #")
        print("############################################\n")
        break

#ricarica i pesi migliori ottenuti prima del blocco/overfitting
if bestweights is not None:
    parametriGNN.load_state_dict(bestweights)


#-----------------------------------------------------
#VALIDATION

parametriGNN.eval()
with torch.no_grad():
    yFORECAST = torch.sigmoid(parametriGNN(xBLOCK, archiIDX))[idxVAL].numpy()

print(f"AUC-ROC: {roc_auc_score(sezioniVAL['target'], yFORECAST):.5f} | AP: {average_precision_score(sezioniVAL['target'], yFORECAST):.5f}")


#-----------------------------------------------------
#probabilità specie per blocco mensile

mese = pd.to_datetime(incidenti["data_iso"]).dt.month
incidentiCOMP = incidenti[(incidenti["data_iso"] >= "2020-01-01") & (incidenti["data_iso"] < start)]     #inizio da quando i dati sono completi

quotaTOT = incidentiCOMP.groupby(["specie", mese]).size().unstack(fill_value=0)     #riga specie e colonna mese
quotaTOT = quotaTOT.div(quotaTOT.sum(axis=1), axis=0)     #ogni riga somma 1

frequenza = incidentiCOMP["specie"].value_counts(normalize=True)     #freq di ogni specie sul totale degli incidenti

#incidenti locale per specie, usa le specie degli incidenti attorno al blocco. se non ci sono incidenti vicini storicamente per X specie uso dati regionali
raggio = 5000     #raggio in cui si conta gli incidenti attorno al blocco
baselineREG = 5     #la frequenza regionale pesa come 5 incidenti, se attorno al blocco ce ne sono pochi uso baseline

#calcola vicini per vedere se influisce su percentuale.

blocchinear = cKDTree(sezioni[["x", "y"]].values).query_ball_point(sezioni[["x", "y"]].values, raggio)     #lista dei blocchi entro raggio

countn = incidentiCOMP.groupby(["blocco", "specie"]).size().unstack(fill_value=0).reindex(index=sezioni.index, columns=frequenza.index, fill_value=0)

incidentinear = np.array([countn.values[v].sum(axis=0) for v in blocchinear])

#formula (incidenti vicini della specie + 5 × frequenza regionale) / (incidenti vicini totali + 5)             5 è baseline per scelta vicini/regione
percincidenti = pd.DataFrame((incidentinear + baselineREG * frequenza.values) / (incidentinear.sum(axis=1, keepdims=True) + baselineREG), index=sezioni.index, columns=frequenza.index)

def probabilita(rischio, b, s, m):     #rischio è probabilità del blocco dal modello, blocco, specie, mese
    tasso = -np.log(1 - np.clip(rischio, 1e-6, 1 - 1e-6)) / ((pd.to_datetime(incidenti["data_iso"]).max() - pd.Timestamp(start)).days / 30.44)      #incidenti attesi al mese sul blocco
    correttivo = 1 if s == "Cinghiale" else 12 * quotaTOT.loc[s, m]  # 12 * quota = 1 se il mese è nella media, il cinghiale ha problemi causa fluttuazione dati quindi manengo 1
    return 1 - np.exp(-tasso * percincidenti.loc[b, s] * correttivo) #probabilità >1 incidente di una specie in quel mese



#-----------------------------------------------------

#TODO confronto con altra NN (da più info? utile? comunque mantenere)
        #specie peso diverso per pericolosità diversa: decidere soglia arbitraria (basata su peso? forse bisogna cambiare la classificazione su R)
        #incidentivicini interesse solo con blocchi senza storia, quindi comunali? se aggiungiamo
        #valutare se modificare target (ora AUC ROC molto alto)
        #prova gnn multiuscita per specie (migliora precisione blocchi, vedi paper)

        #start 2024/06/01

#TODO done
    #importare e provare feature vecchie (tipo meteo? ma se non andava la dubito vada qui)
        #feature vecchie non funzionanti o marginali (spiegate da altre feature già presenti)
    #verificare stagionalità cinghiale
        #correttivo non necessario per fluttuazioni (probabilmnente problema dei dati, la stagionalità c'è, mettere in sviluppi futuri)
    #SC troppo differenti e irrilevanti (centri abitati), rovinano il modello. valutare se farne uno a parte (paper)
        #troppe pochi incidenti e troppa differenza
    #Eliminato le strade comunali per problemi begin end? riaggiungere per completezza
        #strade comunali irrilevanti (principalmente centri abitati)