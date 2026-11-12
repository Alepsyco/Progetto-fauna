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
# TODO Questa parte decentemente
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
validationQUOTA = 0.10     #il test è il resto
start = "2024-06-01"    #per divisione teemporale

specie = ["Cervo", "Cinghiale", "Daino", "Capriolo", "Lupo", "Camoscio", "Tasso", "Sciacallo dorato", "Volpe"]    #specie d'interesse (daino in cervo)
incidenti = incidenti[incidenti["specie"].isin(specie)] #conteggio incidenti per specie

gruppo = incidenti.groupby("pericolosita_specie")["specie"].unique().reindex(["HIGH", "MID", "LOW"]).to_dict() #to_dict per prendere nomi nei gruppi HIGH MID LOW


#------------------------------------------------
#divisione per strada

strada = sezioni["TRIM_SUSTAINER"] + " " + sezioni["TRIM_STR_CODE_1"]     #sempre per distinguere SP1

stradarand = np.random.permutation(strada.unique()) #ordine random per strade (validation e test su starde mai viste per evitare leakage)

splitTRAIN = int(len(stradarand) * trainQUOTA)
splitVAL = int(len(stradarand) * (trainQUOTA + validationQUOTA))

stradaTRAIN = strada.isin(stradarand[:splitTRAIN])     #blocchi del training
stradaVAL = strada.isin(stradarand[splitTRAIN:splitVAL])     #blocchi del validation
stradaTEST = strada.isin(stradarand[splitVAL:])     #blocchi del test

#------------------------------------------------
#divisione incidenti da data e per gruppo HIGH MID LOW (near)

#gli incidenti BF sono input AF sono target

incidentiBF = incidenti[incidenti["data_iso"] < start]
incidentiAF = incidenti[incidenti["data_iso"] >= start]

sezioni["incidentiBF"] = incidentiBF.groupby("blocco").size().reindex(sezioni.index, fill_value=0) #fill_value altrimenti da errore

#divisione da start ma per ogni gruppo di specie (pericolosità HIGH MID LOW)

gruppoBF = incidentiBF.groupby(["blocco", "pericolosita_specie"]).size().unstack(fill_value=0).reindex(index=sezioni.index, columns=list(gruppo), fill_value=0)     #incidenti prima di start (2024/06/01) righe blocchi colonne gruppi
gruppoAF = incidentiAF.groupby(["blocco", "pericolosita_specie"]).size().unstack(fill_value=0).reindex(index=sezioni.index, columns=list(gruppo), fill_value=0)     #incidenti dopo start

for i in gruppo: sezioni["incidentiBF_" + i] = gruppoBF[i]     #incidenti prima di start del gruppo

incidentiCOMP = incidenti[(incidenti["data_iso"] >= "2020-06-01") & (incidenti["data_iso"] < start)]     #inizio da quando i dati sono completi

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

countn.loc[stradaTEST] = 0     #come per near, il test non entra nel conteggio

incidentinear = np.array([countn.values[i].sum(axis=0) for i in blocchinear]) #incidenti vicini a un determinato blocco

#formula (incidenti vicini della specie + 5 × frequenza regionale) / (incidenti vicini totali + 5)             5 è baseline per scelta vicini/regione
percincidenti = pd.DataFrame(
    (incidentinear + baselineREG * frequenza.values) / (incidentinear.sum(axis=1, keepdims=True) + baselineREG),
    index=sezioni.index,
    columns=frequenza.index
)

sezioni["altitudineDEVSTANDARD"] = sezioni["altitudineDEVSTANDARD"].fillna(0)     #NA se il blocco marca solo una cella raster (no devstandard)

#incidenti prima di start entro raggio (5000m) da usare come feature
near = gruppoBF.copy()
near.loc[stradaTEST] = 0    #azzero i blocchi dove strada è dopo splitVAL

grupponear = np.array([near.values[i].sum(axis=0) for i in blocchinear])     #incidentinear per HIGH MID LOW
grupponear = grupponear - near.values     #toglie se stesso

for i, nome in enumerate(gruppo): sezioni["incidentiBFNEAR_" + nome] = np.log1p(grupponear[:, i])     #log per schiacciare i valori molto alti (per HIGH E MID che sono concentrati)


sezioniTRAIN = sezioni[stradaTRAIN]
sezioniTEST = sezioni[stradaTEST]
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
    "incidentiBF",      #incidenti prima di start (BF è BEFORE)
    "incidentiBF_HIGH",   #incidenti prima di start nel blocco (pericolosità HIGH)
    "incidentiBF_MID",
    "incidentiBF_LOW",
    "incidentiBFNEAR_HIGH",   #incidenti prima di start entro raggio (pericolosità HIGH)
    "incidentiBFNEAR_MID",
    "incidentiBFNEAR_LOW"
]



#normalizzazione (media e dev standard solo dal training)
mediaTRAIN = sezioniTRAIN[feature].mean()
devstandardTRAIN = sezioniTRAIN[feature].std() + 1e-9

####################################################################################################################

#parametri

nlayer1 = 32    #neuroni layer
nlayer2 = 16
epochs = 300
learningrate = 0.005
weightdecay = 0.005
lossFN = nn.PoissonNLLLoss()    #loss per output count
dropoutt = 0.3 #spegne casualmente neuroni per limitare overfitting
patience = 100  #per early stopping, n epoche di attesa prima di fermarsi


SAVEWEIGHTS = True #FALSE SE FAI PROVE TRUE ALTRIMENTI
TEST = False    #DA METTERE TRUE SOLO UNA VOLTA ALLA FINE

####################################################################################################################


#-----------------------------------------------------
#NN: per ogni blocco stima numero di incidenti attesi per ogni gruppo HIGH MID LOW

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
        self.nodi1 = nodi1      #per salvataggio torch.save
        self.nodi2 = nodi2      #ditto

        #forma NN (ogni riga ha input e output)
        self.node_emb = nn.Sequential(
            nn.Linear(inputfeature, nodi1),
            nn.ReLU())
        self.gcn = GATv2Conv(inputfeature, nodi1, heads=2, concat=False)     #aggrega feature blocchi vicini

        self.norm = nn.LayerNorm(nodi1)
        self.dropout = nn.Dropout(dropoutt)     #spegne casualmente neuroni per limitare overfitting

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
yBLOCK = torch.tensor(gruppoAF.values, dtype=torch.float)    #target n incidenti a colonna di array 3 valori HIGH MID LOW

idxTRAIN = np.where(stradaTRAIN.values)[0]     #posizioni dei blocchi nei due insiemi
idxVAL = np.where(stradaVAL.values)[0]
idxTEST = np.where(stradaTEST.values)[0]

#-----------------------------------------------------
#train

#DEEP ENSEMBLE: addestro stessa rete con pesi init diversi per mnimizzare errori (vantaggio rispetto a es regressione)
retinum = 3     #reti uguali ma con pesi iniziali diversi, alla fine faccio la media delle previsioni
yMULTI = []     #previsioni di ogni rete
pesiMULTI = []     #pesi di ogni rete

if __name__ == "__main__":     #training poarte solo de eseguito da qui (per import sotto, da vedere)

    for i in range(retinum):
        parametriGNN = RischioGNN(len(feature), nlayer1, nlayer2, len(gruppo))


        patienceCOUNT = 0
        lossBEST = float("inf") #inizializzato con infinito di modo che la prima sia la migliore
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
                previsioneVAL = parametriGNN(xBLOCK, archiIDX)[idxVAL]

            lossEPOCH = lossFN(previsioneVAL, yBLOCK[idxVAL]).item()  #loss validation per epoca

            if lossEPOCH < lossBEST:
                lossBEST = lossEPOCH
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
        pesiMULTI.append(parametriGNN.state_dict())


        parametriGNN.eval() #spegne dropout
        with torch.no_grad():   #senza gradienti
            yMULTI.append(torch.exp(parametriGNN(xBLOCK, archiIDX)).numpy())  #output reti in incidenti attesi (exp: logaritmo a numero) e trasforma in array

    #-----------------------------------------------------
    #VALIDATION

    yFORECAST = np.mean(yMULTI, axis=0)[idxVAL]     #media delle reti blocchi validation usando array dei risultati

    def out(nome, yREALE, yPREVISTO):
        print(f"{nome} | AUC-ROC: {roc_auc_score(yREALE, yPREVISTO):.5f} | AP: {average_precision_score(yREALE, yPREVISTO):.5f}")

    for i, nome in enumerate(gruppo):
        out(nome, yBLOCK[idxVAL, i].numpy() > 0, yFORECAST[:, i])  #yFORECAST ha tre colonne per HIGH MID LOW


    if SAVEWEIGHTS:
            torch.save({
            'model_state_dict': pesiMULTI,     #lista con i pesi di ogni rete
            'feature': feature,
            'gruppo': gruppo,
            'archiIDX': archiIDX,
            'mediaTRAIN': mediaTRAIN,
            'devstandardTRAIN': devstandardTRAIN,
            'start': start,
            'nodi1': parametriGNN.nodi1,
            'nodi2': parametriGNN.nodi2,
            'idxTRAIN': idxTRAIN,
            'stradaTEST': stradaTEST,
            'raggio': raggio,
            'endDATE': incidenti['data_iso'].max()
        }, "weights_strade.pth")

#-----------------------------------------------------
    
    #TEST (DA FARE SOLO UNA VOLTA)
    if TEST is True:

        yFORECAST = np.mean(yMULTI, axis=0)[idxTEST]     #media delle reti blocchi validation usando array dei risultati

        def out(nome, yREALE, yPREVISTO):
            print(f"{nome} | AUC-ROC: {roc_auc_score(yREALE, yPREVISTO):.5f} | AP: {average_precision_score(yREALE, yPREVISTO):.5f}")

        for i, nome in enumerate(gruppo):
            out(nome, yBLOCK[idxTEST, i].numpy() > 0, yFORECAST[:, i])  #yFORECAST ha tre colonne per HIGH MID LOW


'''DA CHIAMARE SU FORECAST (sposto direttamente???)
#probabilità specie per blocco mensile

mese = pd.to_datetime(incidenti["data_iso"]).dt.month   #trasformo stringa in date, dt.month prende mese, mese di ogni incidente
mesiAF = (pd.to_datetime(incidenti["data_iso"]).max() - pd.Timestamp(start)).days / 30.44     #ditto sopra, data più recente - start = n mesi dopo start

quotaTOT = incidentiCOMP.groupby(["specie", mese]).size().unstack(fill_value=0)     #riga specie e colonna mese
quotaTOT = quotaTOT.div(quotaTOT.sum(axis=1), axis=0)     #ogni riga somma 1



def probabilita(rischio, b, s, m):     #incidenti attesi del gruppo s. blocco, specie, mese
    tasso = rischio / mesiAF      #incidenti attesi per gruppo al mese sul blocco
    correttivo = 1 if s in ["Cinghiale", "Lupo", "Camoscio"] else 12 * quotaTOT.loc[s, m]  # (12 * quota = 1) ---> se il mese è nella media. Cinghiale ha problemi causa fluttuazione dati quindi manengo 1
    lista = next(i for i in gruppo.values() if s in i)  #estrae specie di ogni gruppo (oltre HIGH MID LOW)
    percspecie = percincidenti.loc[b, s] / percincidenti.loc[b, lista].sum()     #percentuale della specie specifica
    rischioOUTPUT = 1 - np.exp(-tasso * percspecie * correttivo)  # probabilità >=1 incidenti di una specie in quel mese
    return rischioOUTPUT    #OUTPUT: percentuale di rischio di incidente di XXXXX nel mese YYY: 0.05%
'''


#####################################################################################################################

#TODO confronto con altra NN (da più info? utile? comunque mantenere)
        #benchmark contro storico base (che mesezone non batteva) e altri modelli
        #scegliere cosa fare con camoscio lupo daino (troppi pochi incidenti) INTANTO MESSI IN CORRETTIVO, DA VEDERE
        #possibile simulare dati del traffico basandosi su numero di abitanti e tipo di strada?
        #indagine visiva (maps) per mappare guardrail e barriere in hotspot incidenti (?)


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
        #Multiuscita HIGH MID LOW ok, specie no
    #valutare se modificare target
        #Target numero di incidenti invece che bool (come in MESEZONE)