install.packages("a")
remove.packages("a")

#OPERAZIONI PRELIMINARI
#------------------------------------------------

library("dplyr")
library("tidyverse")
library("extrafont") #Ingnorare 40+ errori generati dalla libreria (developer comment)
library("ggplot2")
library("stringr")
library("sf")     #per openstreetmaps
library("httr")      #Per openmeteo
library("jsonlite") #Per openmeteo
library("terra") #per topografia
#library("lunar")
#library("gapminder")
#library("plotly")
#library("htmlwidgets")
#library("grid")
#library("png")
#library("ggimage")
#library("patchwork")
#library("cowplot")
#library("ggridges")
#library("scales")

#library("conflicted") # Possibile risoluzione per errore
#Unable to find an inherited method for function select for signature "data.frame" (conflitti tra pacchetti)

set.seed(1984)


##################################################################################################################################
# IDEA Analizzo strada, se questa ha avuto incidenti e ha certe caratteristiche la pericolosità di strade simili aumenterà
#
#
##################################################################################################################################
#ZONA PARAMETRI

sogliaSNAP <- 50


##################################################################################################################################

incidenti=read.csv("./incidenti.csv", sep=";")

#rimuovo specie che fanno pochi danni e specie con numero di incidenti non significativo + tutte le entry senza coordinate
incidenti <- incidenti %>% filter(specie %in% c("Capriolo","Cinghiale","Volpe","Tasso","Cervo","Sciacallo dorato","Daino","Lupo","Camoscio"))
incidenti <- incidenti %>% filter(!coordinateWGS84 %in% c(""))

#rimuovo tutte le colonne superflue all'analisi
incidenti <- incidenti %>% select(-any_of(c("altro_personale", "gruppo", "morto_causa","specifica_altra_specie","mancato_recupero")))

#estrazione coordinate lat long per api
incidenti <- incidenti %>% mutate(long = as.numeric(str_extract(coordinateWGS84, "(?<=POINT\\()[-0-9.]+")),
                                  lat = as.numeric(str_extract(coordinateWGS84, "(?<=\\s)[-0-9.]+(?=\\))")))

incidenti <- incidenti %>% mutate(data_iso = format(as.Date(data_intervento, "%Y-%m-%d")),
                                  ora_num = if_else(is.na(ora_segnalamento) | ora_segnalamento == "", as.numeric(substr(ora_intervento, 1, 2)), as.numeric(substr(ora_segnalamento, 1, 2))),
                                  orario = floor(ora_num / 4) * 4)      #MODIFICARE SE SI VUOLE ALTRI BLOCCHI DI ORE, QUI 8

incidenti <- incidenti %>% filter(!is.na(lat), !is.na(long)) %>% st_as_sf(coords = c("long", "lat"), crs = 4326, remove = FALSE)


#classificazione di pericolosità empirica in base al tipo peso e danni dell'animale
incidenti <- incidenti %>% mutate(pericolosita_specie = case_when(
  specie %in% c("Cervo", "Cinghiale", "Daino") ~ "HIGH",
  specie %in% c("Capriolo", "Lupo", "Camoscio") ~ "MID",
  specie %in% c("Tasso", "Sciacallo dorato", "Volpe") ~ "LOW"))

#riempimento dataset generico

incidenti <- incidenti %>% mutate(sesso = if_else(sesso == "" | sesso == "N.D.", "ND", sesso))

incidenti <- incidenti %>% mutate(classe_eta = if_else(classe_eta == "" | classe_eta == "N.D.", "ND", classe_eta))

#accorpamento data e ora per verificare tempo di intervento
incidenti <- incidenti %>% mutate(dataora_segnalamento = as.POSIXct(paste(data_segnalamento, ora_segnalamento), format = "%Y-%m-%d %H:%M:%S"),
                                  dataora_intervento = as.POSIXct(paste(data_intervento, ora_intervento), format = "%Y-%m-%d %H:%M:%S"))

incidenti <- incidenti %>% mutate(dataoraDIFF = as.numeric(difftime(dataora_intervento, dataora_segnalamento, units = "mins")))
incidenti <- incidenti %>% select(-any_of(c("data_segnalamento", "ora_segnalamento", "data_intervento", "ora_intervento")))


#------------------------------------------------

#GETSTRADE da dataset regionale

strade <- parse_url("https://serviziogc.regione.fvg.it/geoserver/RETI_TRASP/wfs")

strade$query <- list(
  service = "WFS",
  version = "2.0.0",
  request = "GetFeature",
  typeName = "RETI_TRASP:GSFVG_IRDAT", #modifica tipo di dato, questo è il più completo
  outputFormat = "application/json",
  srsName = "EPSG:4326"
)

stradeURL <- build_url(strade)
strade <- st_read(stradeURL)


#------------------------------------------------


#da incidenti trova strade vicine alle coordinate e controlla la distanza
stradeNEAR <- st_nearest_feature(incidenti, strade)
stradeDIST <- st_distance(incidenti, strade[stradeNEAR, ], by_element = TRUE)

#Se distanza minore di 50 inserisco la strada (nuova colonna per avere più dati)
incidenti <- incidenti %>% mutate(luogo_segnalamentoCROSS = if_else( as.numeric(stradeDIST) <= sogliaSNAP,
                                                                     strade$TRIM_STR_CODE_1[stradeNEAR],
                                                                     luogo_segnalamento), .before = 11)

incidenti <- st_transform(incidenti, 32633)
strade <- st_transform(strade, 32633)


#unisco archi per ottenere tratti da 1km su cui fare analisi, tolgo SC che non sono ben definite nel dataset

stradeNOSC <- strade %>% filter(CLASSE != "SC")
stradeNOSC <- stradeNOSC %>% mutate(km = floor(((BEGIN + END) / 2) / 1000))     #KM appartenenza arco

#aggiusto lunghezza orig_lenght di 2 NA
stradeNOSC <- stradeNOSC %>% mutate(ORIG_LENGTH = coalesce(ORIG_LENGTH, END - BEGIN))   #coalesce restituisce primo valore non NA che trova, se NA usa calcolo)

#uso TRIM_SUSTAINER per mantenere univoche le strade (es SP 1)
sezioni <- stradeNOSC %>% group_by(TRIM_SUSTAINER, TRIM_STR_CODE_1, CLASSE, km)
sezioni <- sezioni %>% summarise(archi = n(), lunghezza_m = sum(ORIG_LENGTH), do_union = FALSE, .groups = "drop")   #un blocco per strada e chilometro

#snap per distanza e classe strada
incidentiNOSC <- incidenti %>% filter(as.numeric(stradeDIST) <= sogliaSNAP, strade$CLASSE[stradeNEAR] != "SC")    #incidenti su strade extraurbane


sezioni <- sezioni %>% mutate(n_incidenti = tabulate(st_nearest_feature(incidentiNOSC, sezioni), nbins = nrow(sezioni)))#riga del blocco più vicino a incidenti e tabulate conta gli incidenti per blocco

#------------------------------------------------
#GETHABITAT da carta della natura FVG

#se = sensibilità ecologica
#pe = pressione antropica
#ve = valore ecologico
#fg = fragilità ambientale??????????

habitat <- st_read("./CNAT_FVG_2021_v4/CNAT_FVG_2021_v4.shp")

#-------------------------------------------------
#GETALTITUDINE da open topografy
#CITA CON
#Ho, Y.-F., &amp; Hengl, T. (2025). Global ensemble digital terrain model 30m (GEDTM30) (Version v20250619) [Data set]. Distributed by OpenTopography. https://doi.org/10.5069/G9BV7DT1. Accessed 2026-09-01

FVGelevation <- rast("FVGelevation.tif")


#-------------------------------------------------
#calcolo area bosco e case vicino

#SE SI VUOLE CAMBIARE SOGLIE BOSCO MODIFICA SOTTO E CANCELLA SEZIONI RDS

if (file.exists("sezioni.rds")) {
  sezioni <- readRDS("sezioni.rds")
} else {
  
  
  boschi <- st_read("./CNAT_FVG_2021_v4/CNAT_FVG_2021_v4.shp") %>% filter(grepl("^(41|42|44)", CODICE)) #filtro habitat seleziono solo bosco
  boschi <- boschi %>% st_transform(32633) %>% st_make_valid() #cs corretto + aggiustamento poligoni
  boschi <- boschi %>% st_union()
  
  sezioni <- sezioni %>% mutate(id = 1:nrow(sezioni))   #serve per riassegnare le quote ai blocchi giusti dopo l'intersezione
  
  #dato raggio calcola la quota di bosco
  bosco <- function(raggio) {
    buffer <- st_buffer(sezioni, raggio)     #st_buffer allarga poligono di raggio metri e restituisce un poligono per blocco
    inters <- suppressWarnings(st_intersection(buffer, boschi))     #st_intersection ritaglia buffer sul bosco (no err)
    quota <- rep(0, nrow(sezioni))     #vettore 0 lungo quanto n sezioni
    quota[inters$id] <- st_area(inters) / st_area(buffer)[inters$id]     #area bosco / area buffer
    quota
  }
  
  sezioni <- sezioni %>% mutate(bosco20 = bosco(20))      #quota di bosco entro 20 m dal blocco
  sezioni <- sezioni %>% mutate(bosco200 = bosco(200))
  sezioni <- sezioni %>% mutate(bosco500 = bosco(500))
  
  saveRDS(sezioni, "sezioni.rds")
  
}

#edifici
edifici <- st_read("./06_Friuli-Venezia_Giulia-2026-07-31T05Z.osm.pbf", query = "SELECT building FROM multipolygons WHERE building IS NOT NULL")
edifici <- edifici %>% st_transform(32633) %>% st_centroid() #uso centroid per contare edifici una sola volta (gruppi uniti, città paesi)

sezioni <- sezioni %>% mutate(edifici50 = lengths(st_intersects(st_buffer(sezioni, 50), edifici)))   #ditto come per bosco, lenghts conta quanti poligoni trovo
sezioni <- sezioni %>% mutate(edifici200 = lengths(st_intersects(st_buffer(sezioni, 200), edifici)))

#altitudine
#vect converte sezioni da sf a terra e project li rende compatibili con raster del file (crs =), extract fa media celle toccate dai blocchi e [[1]] output vettore
sezioni <- sezioni %>% mutate(altitudineMEAN = extract(FVGelevation, project(vect(sezioni), crs(FVGelevation)), fun = mean, na.rm = TRUE, ID = FALSE)[[1]])

#ditto per devstandard, misura differenza di quote attarversate dal blocco (sarà NA se tocca solo 1 visto che non c'è differenza)
sezioni <- sezioni %>% mutate(altitudineDEVSTANDARD = extract(FVGelevation, project(vect(sezioni), crs(FVGelevation)), fun = sd, na.rm = TRUE, ID = FALSE)[[1]])


#coordinate centroide blocco
sezioni <- sezioni %>% mutate(x = st_coordinates(st_centroid(st_geometry(sezioni)))[, 1], y = st_coordinates(st_centroid(st_geometry(sezioni)))[, 2])   #[, 1] e [, 2] per X e Y



#-------------------------------------------------
#-------------------------------------------------

#DATASET FINALE SU CUI OPERIAMO (sezioni e incidenti legati ai blocchi calcolati)


write.csv(sezioni %>% st_drop_geometry(), "sezioni.csv")   

incidentiBLOCCHI <- incidentiNOSC %>% mutate(blocco = st_nearest_feature(incidentiNOSC, sezioni)) %>% st_drop_geometry() #colonna blocco più vicino a incidente
incidentiBLOCCHI <- incidentiBLOCCHI %>% select(blocco, data_iso, specie) 

write.csv(incidentiBLOCCHI, "incidentiBLOCCHI.csv", row.names = FALSE)     #un incidente per riga con il blocco a cui appartiene

###################################################################
###################################################################

