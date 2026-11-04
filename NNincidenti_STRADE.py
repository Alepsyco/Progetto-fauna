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
#
#start 2024/06/01
######################################################################################################################################

#Carica dataset finale da R

sezioni = pd.read_csv("sezioni.csv", index_col=0)     #un blocco per riga, l'indice è il numero di riga del blocco (da R, parte da 1)
incidenti = pd.read_csv("incidentiBLOCCHI.csv")       #un incidente per riga con il numero del blocco a cui appartiene

torch.manual_seed(1984)
np.random.seed(1984)


#------------------------------------------------
#specie e gruppi e parametri divisione

trainQUOTA = 0.80
valdationQUOTA = 0.10     #il test è il resto
start = "2024-06-01"    #per divisione teemporale

specie = ["Cervo", "Cinghiale", "Daino", "Capriolo", "Lupo", "Camoscio", "Tasso", "Sciacallo dorato", "Volpe"]    #specie d'interesse
incidenti = incidenti[incidenti["specie"].isin(specie)] #conteggio incidenti

gruppo = incidenti.groupby("pericolosita_specie")["specie"].unique().reindex(["HIGH", "MID", "LOW"]).to_dict() #to_dict per prendere nomi nei gruppi HIGH MID LOW

#------------------------------------------------
#divisione incidenti da data e per specie (near)

#gli incidenti BF sono parte del training AF sono target

incidentiBF = incidenti[incidenti["data_iso"] < start]
incidentiAF = incidenti[incidenti["data_iso"] >= start]

sezioni["incidentiBF"] = incidentiBF.groupby("blocco").size().reindex(sezioni.index, fill_value=0) #fill_value altrimenti da errore

#divisione da start ma per ogni gruppo di specie (pericolosità HIGH MID LOW)
gruppoBF = incidentiBF.groupby(["blocco", "pericolosita_specie"]).size().unstack(fill_value=0).reindex(index=sezioni.index, columns=list(gruppo), fill_value=0)     #incidenti prima di start (2024/06/01) righe blocchi colonne gruppi
gruppoAF = incidentiAF.groupby(["blocco", "pericolosita_specie"]).size().unstack(fill_value=0).reindex(index=sezioni.index, columns=list(gruppo), fill_value=0)     #incidenti dopo start

for i in gruppo: sezioni["incidentiBF_" + i] = gruppoBF[i]     #incidenti prima di start del gruppo

incidentiCOMP = incidenti[(incidenti["data_iso"] >= "2020-01-01") & (incidenti["data_iso"] < start)]     #inizio da quando i dati sono completi

frequenza = incidentiCOMP["specie"].value_counts(normalize=True)     #freq di ogni specie sul totale degli incidenti

#incidenti locale per specie, usa le specie degli incidenti attorno al blocco. se non ci sono incidenti vicini storicamente per X specie uso storico regionale
raggio = 5000     #raggio in cui si conta gli incidenti attorno al blocco
baselineREG = 5     #la frequenza regionale pesa come 5 incidenti, se attorno al blocco ce ne sono pochi uso baseline

#calcola vicini (entro 5000m) per vedere se influisce su percentuale.

blocchinear = (
    cKDTree(sezioni[["x", "y"]].values) #trova vicini in base a coordinate
    .query_ball_point(sezioni[["x", "y"]].values, raggio)   #crea lista
)

countn = (
    incidentiCOMP.groupby(["blocco", "specie"]).size()  #count incidenti per coppia blocco,specie
    .unstack(fill_value=0).reindex(index=sezioni.index, columns=frequenza.index, fill_value=0)  #aggiunge anche blocchi senza incidenti (valore 0)
)

incidentinear = np.array([countn.values[i].sum(axis=0) for i in blocchinear]) #incidenti vicini a un determinato blocco

#formula (incidenti vicini della specie + 5 × frequenza regionale) / (incidenti vicini totali + 5)             5 è baseline per scelta vicini/regione
percincidenti = pd.DataFrame(
    (incidentinear + baselineREG * frequenza.values) / (incidentinear.sum(axis=1, keepdims=True) + baselineREG),
    index=sezioni.index,
    columns=frequenza.index
)

sezioni["altitudineDEVSTANDARD"] = sezioni["altitudineDEVSTANDARD"].fillna(0)     #NA se il blocco marca solo una cella raster (no devstandard)

#------------------------------------------------
#divisione per strada

strada = sezioni["TRIM_SUSTAINER"] + " " + sezioni["TRIM_STR_CODE_1"]     #sempre per distinguere SP1

stradarand = np.random.permutation(strada.unique()) #ordine random per strade

splitTRAIN = int(len(stradarand) * trainQUOTA)
splitVAL = int(len(stradarand) * (trainQUOTA + valdationQUOTA))

sezioniTRAIN = sezioni[strada.isin(stradarand[:splitTRAIN])]
sezioniTEST = sezioni[strada.isin(stradarand[splitVAL:])]

#------------------------------------------------
#FEATURE PER NODO

feature = [
    "altitudineMEAN",          #altitudine media blocco
    "altitudineDEVSTANDARD",  #differenza altitudine per blocco
    "bosco20",                #percentuale bosco in 20 metri attorno
    "bosco200",                #"" 200 metri
    "bosco500",                #"" 500 metri
    "edifici50",               #edifici, 50 metri
    "edifici200",               #"" 200 metri
    "lunghezza_m",             #metri strada
    "incidentiBF",          #incidenti prima di start (BF è BEFORE)
    "incidentiBF_HIGH",   #incidenti prima di start (pericolosità HIGH)
    "incidentiBF_MID",
    "incidentiBF_LOW"
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
#NN: per ogni blocco stima il rischio dei tre gruppi HIGH MID LOW

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
    def __init__(self, inputfeature, nodi1, nodi2, uscite):
        super().__init__()

        #forma NN (ogni riga ha input e output)
        self.node_emb = nn.Sequential(
            nn.Linear(inputfeature, nodi1),
            nn.ReLU())
        self.gcn = GATv2Conv(inputfeature, nodi1, heads=2, concat=False)     #aggrega feature blocchi vicini

        self.norm = nn.LayerNorm(nodi1)
        self.dropout = nn.Dropout(0.3)     #spegne casualmente neuroni per limitare overfitting

        self.output = nn.Sequential(        #output riceve il risultato più feature e dà 3 uscite
            nn.Linear(nodi1 + inputfeature, nodi2),
            nn.ReLU(),
            nn.Linear(nodi2, uscite)
        )

#funzione NN
    def forward(self, xBLOCK, edgeIDX):
        gcn_out = self.gcn(xBLOCK, edgeIDX)
        local_out = self.node_emb(xBLOCK)

        valori = torch.relu(self.norm(gcn_out + local_out)) #res connection (vicini + feature del blocco)
        valori = self.dropout(valori)

        valori = torch.cat([valori, xBLOCK], dim=-1)

        rischio = self.output(valori)
        return rischio

#grafo di blocchi
xBLOCK = torch.tensor(((sezioni[feature] - mediaTRAIN) / devstandardTRAIN).values, dtype=torch.float)
yBLOCK = torch.tensor((gruppoAF > 0).values, dtype=torch.float)    #target a colonna di array 3 valori HIGH MID LOW

idxTRAIN = np.where(strada.isin(stradarand[:splitTRAIN]).values)[0]     #posizioni dei blocchi nei due insiemi
idxVAL = np.where(strada.isin(stradarand[splitTRAIN:splitVAL]).values)[0]

#-----------------------------------------------------

#train

parametriGNN = RischioGNN(len(feature), 16, 8, len(gruppo))

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

def out(nome, yREALE, yPREVISTO):
    print(f"{nome} | AUC-ROC: {roc_auc_score(yREALE, yPREVISTO):.5f} | AP: {average_precision_score(yREALE, yPREVISTO):.5f}")

for i, nome in enumerate(gruppo):
    out(nome, yBLOCK[idxVAL, i].numpy(), yFORECAST[:, i])     #yFORECAST ha tre colonne per HIGH MID LOW

#-----------------------------------------------------
#probabilità specie per blocco mensile

mese = pd.to_datetime(incidenti["data_iso"]).dt.month

quotaTOT = incidentiCOMP.groupby(["specie", mese]).size().unstack(fill_value=0)     #riga specie e colonna mese
quotaTOT = quotaTOT.div(quotaTOT.sum(axis=1), axis=0)     #ogni riga somma 1


def probabilita(rischio, b, s, m):     #rischio è probabilità del gruppo s. blocco, specie, mese
    tasso = -np.log(1 - np.clip(rischio, 1e-6, 1 - 1e-6)) / ((pd.to_datetime(incidenti["data_iso"]).max() - pd.Timestamp(start)).days / 30.44)      #incidenti attesi per gruppo al mese sul blocco
    correttivo = 1 if s == "Cinghiale" else 12 * quotaTOT.loc[s, m]  # 12 * quota = 1 se il mese è nella media, il cinghiale ha problemi causa fluttuazione dati quindi manengo 1
    lista = next(i for i in gruppo.values() if s in i)  #estrae specie di ogni gruppo (oltre HIGH MID LOW)
    return 1 - np.exp(-tasso * percincidenti.loc[b, s] / percincidenti.loc[b, lista].sum() * correttivo) #probabilità >1 incidenti di una specie in quel mese



#####################################################################################################################

#TODO confronto con altra NN (da più info? utile? comunque mantenere)
        #valutare se modificare target (ora AUC ROC molto alto)
        #benchmark contro storico base (che mesezone non batteva) e altri modelli
        #scegliere cosa fare con camoscio lupo daino (troppi pochi incidenti)
        #possibile simulare dati del traffico basandosi su numero di abitanti e tipo di strada?


#TODO done
    #importare e provare feature vecchie (tipo meteo? ma se non andava la dubito vada qui)
        #feature vecchie non funzionanti o marginali (spiegate da altre feature già presenti)
    #verificare stagionalità cinghiale
        #correttivo non necessario per fluttuazioni (probabilmnente problema dei dati, la stagionalità c'è, mettere in sviluppi futuri)
    #SC troppo differenti e irrilevanti (centri abitati), rovinano il modello. valutare se farne uno a parte (paper)
        #troppe pochi incidenti e troppa differenza
    #Eliminato le strade comunali per problemi begin end? riaggiungere per completezza
        #strade comunali irrilevanti (principalmente centri abitati)
    #prova gnn multiuscita per specie (migliora precisione blocchi, vedi paper)