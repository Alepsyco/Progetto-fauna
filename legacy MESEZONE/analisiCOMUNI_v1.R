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

#in incidenti, trova strade vicine alle coordinate e controlla la distanza
stradeNEAR <- st_nearest_feature(incidenti, strade)
stradeDIST <- st_distance(incidenti, strade[stradeNEAR, ], by_element = TRUE)

#Se distanza minore di 20 inserisco la strada (nuova colonna per avere più dati)
incidenti <- incidenti %>% mutate(luogo_segnalamentoCROSS = if_else( as.numeric(stradeDIST) <= 20, 
                                                                strade$TRIM_STR_CODE_1[stradeNEAR], 
                                                                luogo_segnalamento), .before = 11)
#------------------------------------------------


#GETMETEO da openmeteo

getmeteo <- function(lat, lon, data, orario) {
  res <- GET(
    "https://archive-api.open-meteo.com/v1/archive",
    query = list( 
      latitude = lat,
      longitude = lon,
      start_date = data,
      end_date = data,
      hourly = "temperature_2m,precipitation,windspeed_10m",
      timezone = "Europe/Rome"
    )
  )

  if (status_code(res) != 200) {
    return(tibble(lat = NA, long = NA, data_iso = NA, precipitazioni = NA)) #fallback a NA e seleziona cosa scegliere cosa chiedere a API
  }
  
datiM <- fromJSON(content(res, as = "text", encoding = "UTF-8"))

  
indiceORA <- orario + 1 #array orario parte da index 1 mezzanotte
  
tibble(
  lat = lat,
  long = lon,
  data_iso = data,
  orario = orario,
  temp = datiM$hourly$temperature_2m[indiceORA],
  precipitazioni = datiM$hourly$precipitation[indiceORA],
  vento = datiM$hourly$windspeed_10m[indiceORA])}  

#FUNZIONE CHIAMATA API SU TUTTO DATASET TOKEN LIMITATIN E LENTO ----NON ESEGUIRE A CASO----
  
#meteo <- pmap_dfr(incidenti %>% st_drop_geometry() %>% select(lat, long, data_iso, orario),     #dropgeometry se no si rompe
#                  function(lat, long, data_iso, orario) {
#                    Sys.sleep(0.1) #pausa per non rompere api
#                    getmeteo(lat, long, data_iso, orario)
# })

#incidenti <- incidenti %>% bind_cols(meteo %>% select(temp, precipitazioni, vento))

meteo <- read.csv2("meteoincidenti.csv", sep = ",", header = TRUE) #SE PER SBAGLIO RESETTI INCIDENTI LEGGI DA QUA


#-------------------------------------------------

##GETCOMUNI da dataset regionale

comuni <- parse_url("https://serviziogc.regione.fvg.it/geoserver/UNITA_AMM/wfs")

comuni$query <- list(
  service = "WFS",
  version = "2.0.0",
  request = "GetFeature",
  typeName = "UNITA_AMM:COMUNI_FVG", 
  outputFormat = "application/json",
  srsName = "EPSG:4326"
)

comuniURL <- build_url(comuni)
comuni <- st_read(comuniURL)
comuni <- comuni %>% st_make_valid() #non si può operare sopra se non si fa questo a causa di poligoni non validi

#rimuovo tutte le colonne superflue all'analisi 
comuni <- comuni %>% select(-any_of(c("id", "ID_OGG", "COD_ISTAT","COD_CATASTALE")))

incidenti <- incidenti %>% st_join(comuni %>% select(NOME), join = st_within)


#-------------------------------------------------

#GETHABITAT da carta della natura FVG

#se = sensibilità ecologica
#pe = pressione antropica
#ve = valore ecologico
#fg = fragilità ambientale??????????

habitat <- st_read("./CNAT_FVG_2021_v4/CNAT_FVG_2021_v4.shp")

habitat <- st_transform(habitat, crs = st_crs(comuni)) #standardizza crs per join

sf_use_s2(FALSE)  # toglie spherical geometry (?) riduce di molto il tempo per l'intersection
habitatJOIN <- st_intersection(comuni %>% select(ID, NOME), habitat %>% select(classe_ve, classe_se, classe_pa))
sf_use_s2(TRUE)


#cambio da ordinale a numerico per media + calcolo area per comune
levels <- c("Molto bassa" = 1, "Bassa" = 2, "Media" = 3, "Alta" = 4, "Molto alta" = 5)
habitatJOIN <- habitatJOIN %>% mutate(senseco = levels[classe_se],
                                      pressantro = levels[classe_pa],
                                      valeco = levels[classe_ve])

habitatJOIN <- habitatJOIN %>% select(-any_of(c("classe_se", "classe_pa", "classe_ve")))

#area dell'habitat joinato
habitatJOIN <- habitatJOIN %>% mutate(area = st_area(.))

#media dei valori sopra, da 1 minimo a 5 massimo
habitatJOIN <- habitatJOIN %>% group_by(ID,NOME) %>% summarise(sensecoM = weighted.mean(senseco, w = as.numeric(area), na.rm = TRUE),
                                                              pressantroM = weighted.mean(pressantro, w = as.numeric(area), na.rm = TRUE),
                                                              valecoM = weighted.mean(valeco, w = as.numeric(area), na.rm = TRUE))

#matrice di adiacenza tra comuni 
adjMATRIX <- st_touches(comuni, sparse = FALSE)  
rownames(adjMATRIX) <- comuni$NOME
colnames(adjMATRIX) <- comuni$NOME

#-------------------------------------------------
#GETALTITUDINE da open topografy
#CITA CON
#Ho, Y.-F., &amp; Hengl, T. (2025). Global ensemble digital terrain model 30m (GEDTM30) (Version v20250619) [Data set]. Distributed by OpenTopography. https://doi.org/10.5069/G9BV7DT1. Accessed 2026-09-01

#FVGelevation.tif è un quadrato che comprende anche territori esterni
FVGelevation <- rast("FVGelevation.tif")

#statistiche
global(
  FVGelevation,
  fun = c("min", "max", "mean", "sd"),
  na.rm = TRUE
)

comuniTERRA <- vect(comuni)  #converte in formato library terra per aggiustare i confini

altitudineFVG <- crop(FVGelevation, comuniTERRA)
altitudineFVG <- mask(altitudineFVG, comuniTERRA)  #elimina territori fuori dai confini dei comuni FVG 

#media altitudine per comune (rename necessario causa funzionamento funzioni library terra che da problemi nel bind_cols)
altitudineMEAN <- extract(altitudineFVG, comuniTERRA, fun = mean, na.rm = TRUE, ID = FALSE) %>% rename(altitudineMEAN = FVGelevation)

#deviazione standard altitudine per comune
altitudineDEVSTANDARD <- extract(altitudineFVG, comuniTERRA, fun = sd, na.rm = TRUE, ID = FALSE) %>% rename(altitudineDEVSTANDARD = FVGelevation)

altitudineCOMUNI <- bind_cols(NOME = comuni$NOME, altitudineMEAN, altitudineDEVSTANDARD)


#-------------------------------------------------
#-------------------------------------------------


#####################################################
#V2
#MODIFICATO SCOPE DA SETTIMANALE A MENSILE. PER SETTIMANA LA PREVISIONE ERA TROPPO GENERICA.
#####################################################


incidenti <- incidenti %>% mutate(mese = format(as.Date(data_iso), "%Y-%m"))

incidentimese <- incidenti %>% st_drop_geometry() %>% count(NOME, mese, name = "n_incidenti")
incidentimese <- incidentimese %>% left_join(comuni %>% st_drop_geometry() %>% select(ID, NOME), by = "NOME")

#griglia comune + mese inclusi 0 incidenti, per vedere esempi negativi

inizio <- min(as.Date(incidenti$data_iso), na.rm = TRUE)
fine   <- max(as.Date(incidenti$data_iso), na.rm = TRUE)

meseGRID <- format(seq(from = as.Date(cut(inizio, "month")), 
                       to = as.Date(cut(fine, "month")), 
                       by = "month"), "%Y-%m")

comuniGRID <- unique(comuni$NOME)

#creo griglia di previsione con target per confronto con previsione in training 
griglia <- expand_grid(NOME = comuniGRID, mese = meseGRID)

incidentimese <- griglia %>% left_join(incidentimese %>% select(-any_of("ID")), by = c("NOME", "mese"))
incidentimese <- incidentimese %>% mutate(n_incidenti = replace_na(n_incidenti, 0),target = if_else(n_incidenti > 0, 1, 0))

#Ho bisogno di un punto rappresentativo per il meteo comunale della settimana
#Provato centro del comune ma vengono fuori troppe chiamate

comuniCENTRO <- st_centroid(comuni) %>% mutate(lon = st_coordinates(.)[,1], lat = st_coordinates(.)[,2]) %>% st_drop_geometry() 
comuniCENTRO <- comuniCENTRO %>% select(NOME, lat, lon)

#creo tot punti a cui fanno riferimento i vari comuni (da ~250 a 25, per meteo)
comuniAGG <- kmeans(comuniCENTRO %>% select(lat, lon), centers = 25)

comuniCENTRO <- comuniCENTRO %>% mutate(zona = comuniAGG$cluster)

zonaCENTRO <- as.data.frame(comuniAGG$centers) %>% mutate(zona = row_number())


#numero del mese per encoding ciclico stagionalità (usato poi in py al posto di weeknum)

incidentimese <- incidentimese %>% mutate(monthnum = as.numeric(substr(mese, 6, 7)))

#per osservare incidenti settimane precedenti senza inquinare dati
incidentimese <- incidentimese %>% arrange(NOME, mese) %>% group_by(NOME) %>% mutate(n_incidenti_PREV = lag(n_incidenti, default = 0)) %>% ungroup()


getmeteoZONE <- function(mese) {
  start <- as.Date(paste0(mese, "-01"))
  end <- start + months(1) - 1
  
  res <- GET(
    "https://archive-api.open-meteo.com/v1/archive",
    query = list(
      latitude = paste(zonaCENTRO$lat, collapse = ","),
      longitude = paste(zonaCENTRO$lon, collapse = ","),
      start_date = format(start),
      end_date = format(end),
      daily = "temperature_2m_mean,precipitation_sum,windspeed_10m_max",
      timezone = "Europe/Rome"
    )
  )
  
  if (status_code(res) != 200) return(NULL)
  
  datiM2 <- fromJSON(content(res, as = "text", encoding = "UTF-8"))
  
  map_dfr(seq_along(zonaCENTRO$zona), function(i) {
    tibble(
      zona = zonaCENTRO$zona[i],
      mese = mese,
      tempMEAN = mean(datiM2$daily$temperature_2m_mean[[i]], na.rm = TRUE),
      precipMAX = max(datiM2$daily$precipitation_sum[[i]], na.rm = TRUE),
      ventoMEAN = mean(datiM2$daily$windspeed_10m_max[[i]], na.rm = TRUE),
      precipMEAN = mean(datiM2$daily$precipitation_sum[[i]], na.rm = TRUE),
      precipSOGLIA = sum(datiM2$daily$precipitation_sum[[i]] > 10, na.rm = TRUE)
      
    )
  })
}


#CHIAMATA API (limitata no problem)
if (file.exists("meteoZONE_FINAL_V2.rds")) {
  meteoZONE_mese <- readRDS("meteoZONE_FINAL_V2.rds")
} else {
  meteoZONE_mese <- tibble() #se non esiste creo tabella vuota
}

mesiMANCANTI <- setdiff(mesiUNICI, unique(meteoZONE_mese$mese))

for (m in mesiMANCANTI) {
  Sys.sleep(0.8)
  meteoZONE_mese <- bind_rows(meteoZONE_mese, getmeteoZONE(m))
  saveRDS(meteoZONE_mese, "meteoZONE_FINAL_V2.rds")
}

#meteoZONE_mese <- read.csv2("meteoZONE_FINAL_V2.rds", sep = ",", header = TRUE) #SE PER SBAGLIO RESETTI LEGGI DA QUA (old)
meteoCOMUNI <- comuniCENTRO %>% select(NOME, zona) 
meteoCOMUNI <- meteoCOMUNI %>% left_join(meteoZONE_mese, by = "zona")

#matrice di distanza tra centri comunali, permette FORSE di aggiungere un layer alla NN e guardare vicini dei vicini
distMATRIX <- as.matrix(dist(comuniCENTRO %>% select(lon, lat))) * 111 #moltiplico per passare da lat/lon a km

#peso archi = 1/distanza
pesiMATRIX <- (1 / pmax(distMATRIX, 0.1)) * adjMATRIX

dimnames(pesiMATRIX) <- dimnames(adjMATRIX)
diag(pesiMATRIX) <- 0 #non considero se stessi


#-------------------------------------------------
#densità strade per zona


sf_use_s2(FALSE) # toglie spherical geometry (?) riduce di molto il tempo per l'intersection
stradeJOIN <- st_intersection(comuni %>% select(ID, NOME), strade %>% select(CLASSE))
sf_use_s2(TRUE)

stradeJOIN <- stradeJOIN %>% mutate(km = as.numeric(st_length(.)) / 1000)

#km totali di strada per comune 
stradeTOT <- stradeJOIN %>% st_drop_geometry() %>% group_by(NOME)
stradeTOT <- stradeTOT %>% summarise(strade_km = sum(km, na.rm = TRUE))
  
#densità = km di strada totali / km^2 comune, normalizza per dimensione del comune
comuniAREA <- comuni %>% mutate(AREA_KQ = AREA_MQ / 1000000) %>% st_drop_geometry() 
comuniAREA <- comuniAREA %>% select(NOME, AREA_KQ, AREA_MQ, PERIM_ML, ID) %>% distinct(NOME, .keep_all = TRUE)


stradeTOT <- stradeTOT %>% left_join(comuniAREA, by = "NOME")
stradeTOT <- stradeTOT %>% mutate(density = strade_km / AREA_KQ)   #densità strade
stradeTOT <- stradeTOT %>% select(NOME, strade_km, density, AREA_KQ)

#distinzione per tipo di strada
stradeCLASSE <- stradeJOIN %>% st_drop_geometry()
stradeCLASSE <- stradeCLASSE %>% group_by(NOME, CLASSE)
stradeCLASSE <- stradeCLASSE %>% summarise(km = sum(km, na.rm = TRUE)) #km totali
stradeCLASSE <- stradeCLASSE %>% pivot_wider(names_from = CLASSE, values_from = km, values_fill = 0)  #elementi di CLASSE strade diventano singole colonne (conta km totali per tipo di strada)


#accorpo classi di strada inutili
stradeCLASSE <- stradeCLASSE %>% mutate(AUT = replace_na(AS, 0) + replace_na(RA, 0),
                                        SPRI = replace_na(SS, 0) + replace_na(SR, 0) + replace_na(NSA, 0),
                                        SLOC = replace_na(SP, 0),
                                        SCOM = replace_na(SC, 0))

stradeCLASSE <- stradeCLASSE %>% select(NOME, AUT, SPRI, SLOC, SCOM)



#-------------------------------------------------
#-------------------------------------------------



#DATASET FINALE SU CUI OPERIAMO (habitat per comune + meteo per comune + incidenti per comune con target di previsione)

#incidenti e meteo
datasetNODES <- incidentimese %>% left_join(habitatJOIN %>% st_drop_geometry(), by = c("NOME"))
datasetNODES <- datasetNODES %>% left_join(meteoCOMUNI, by = c("NOME", "mese"))

#densità totale e tipi strada
datasetNODES <- datasetNODES %>% left_join(stradeTOT, by = "NOME")
datasetNODES <- datasetNODES %>% left_join(stradeCLASSE, by = "NOME") 
datasetNODES <- datasetNODES %>% mutate(across(c(AUT, SPRI, SLOC, SCOM)))

#SCOM da solo non è utile alla NN, provo quantità di strade comunali su tot e su area comune
datasetNODES <- datasetNODES %>% mutate(SCOM_percentage = replace_na(SCOM / strade_km, 0))
datasetNODES <- datasetNODES %>% mutate(SCOM_density = SCOM / AREA_KQ)

#altitudine comunale (media e dev standard)
datasetNODES <- datasetNODES %>% left_join(altitudineCOMUNI, by = "NOME")

write.csv(as.data.frame(pesiMATRIX), "matricePESI.csv")
write.csv(datasetNODES, "datasetNODES_V3.csv", row.names = FALSE)
write.csv(as.data.frame(adjMATRIX), "matriceADJ.csv")



#-------------------------------------------------
#-------------------------------------------------




#PLOT VARI

#habitat generale + incidenti
ggplot() +
  geom_sf(data = habitat, aes(fill = nomeclasse), color = NA) +
  geom_sf(data = incidenti, color = "red", size = 0.3, alpha = 0.6) +
  labs(
    title = "Habitat con overlay degli incidenti",
    subtitle = "Tipo Habitat"
  )+
  theme(
    text = element_text(size = 10, family = "Helvetica"),
    legend.position = "none"
  )


#habitat pressione antropica + incidenti
ggplot() +
  geom_sf(data = habitat, aes(fill = classe_pa), color = NA) +
  geom_sf(data = incidenti, color = "red", size = 0.3, alpha = 0.6) +
  labs(
    title = "Habitat con overlay degli incidenti",
    subtitle = "Pressione Antropica",
    fill = "Pressione Antropica"
  )+
  theme(
    text = element_text(size = 10, family = "Helvetica"),
  )


#strade + incidenti

stradePLOT <- st_zm(strade, drop = TRUE) #droppa dimensioni oltre a XY (altrimenti non plotta con incidenti)

ggplot() +
  geom_sf(data = stradePLOT) +
  geom_sf(data = incidenti, color = "red", size = 0.3, alpha = 0.6) +
  labs(
    title = "Strade FVG con overlay incidenti",
    subtitle = ""
  )+
  theme(
    text = element_text(size = 10, family = "Helvetica"),
    legend.position = "none"
  )



#comuni + incidenti ()
ggplot() +
  geom_sf(data = comuni, color = "black", color = NA) +
  geom_sf(data = incidenti, color = "red", size = 0.3, alpha = 0.6) +
  labs(
    title = "Comuni con overlay incidenti",
    subtitle = ""
  )+
  theme(
    text = element_text(size = 10, family = "Helvetica"),
    legend.position = "none"
  )




###################################################################
###################################################################

#TODO

#AGGIUSTARE mesiUNICI

