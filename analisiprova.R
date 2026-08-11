install.packages("lunar")
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
library("lunar")
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

incidenti <- incidenti %>% mutate(classe_eta = if_else(classe_eta == "" , "ND", sesso))

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

#se ho il luogo vuoto e distanza minore di 30 inserisco la strada
incidenti <- incidenti %>% mutate(luogo_segnalamentoCROSS = if_else( as.numeric(stradeDIST) <= 20, 
                                                                strade$TRIM_STR_CODE_1[stradeNEAR], 
                                                                luogo_segnalamento), .before = 11)
#------------------------------------------------

#GETLUNA da libreria
#Full = luna piena
#Waning = luna calante
#Waxing = luna crescente
#New = luna nuova
#in alternativa name = FALSE per valori in gradi (provare se c'è correlazione?) 

incidenti <- incidenti %>% mutate(fase_lunare = lunar.phase(as.Date(data_iso), name = TRUE))

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
      hourly = "temperature_2m,temperature_2m,precipitation,windspeed_10m",
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
  
meteo <- pmap_dfr(incidenti %>% st_drop_geometry() %>% select(lat, long, data_iso, orario),     #dropgeometry se no si rompe
                  function(lat, long, data_iso, orario) {
                    Sys.sleep(0.1) #pausa per non rompere api
                    getmeteo(lat, long, data_iso, orario)
  }
)

incidenti <- incidenti %>% bind_cols(meteo %>% select(temp, precipitazioni, vento))

#meteo <- read.csv2("meteoincidenti.csv", header = TRUE) #SE PER SBAGLIO RESETTI INCIDENTI LEGGI DA QUA


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
#fg = ????????????

habitat <- st_read("./CNAT_FVG_2021_v4/CNAT_FVG_2021_v4.shp")

habitat <- st_transform(habitat, crs = st_crs(comuni)) #standardizza crs per join

sf_use_s2(FALSE)  # toglie spherical geometry (?) riduce di molto il tempo per l'intersection
habitatJOIN <- st_intersection(comuni %>% select(NOME), habitat %>% select(classe_ve, classe_se, classe_pa))
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
habitatJOIN <- habitatJOIN %>% group_by(NOME) %>% summarise(sensecoM = weighted.mean(senseco, w = as.numeric(area), na.rm = TRUE),
                                                             pressantroM = weighted.mean(pressantro, w = as.numeric(area), na.rm = TRUE),
                                                             valecoM = weighted.mean(valeco, w = as.numeric(area), na.rm = TRUE))

#matrice di adiacenza tra comuni 
adjMATRIX <- st_touches(comuni, sparse = FALSE)  
rownames(adjMATRIX) <- comuni$NOME
colnames(adjMATRIX) <- comuni$NOME

#-------------------------------------------------

#aggiustamenti per training

#crea colonna settimane (per training)
incidenti <- incidenti %>% mutate(settimana = format(as.Date(data_iso), "%Y-%V"))
incidentiweek <- incidenti %>% st_drop_geometry() %>% count(NOME, settimana, name = "n_incidenti")

#griglia comune + settimana inclusi 0 incidenti, per vedere esempi negativi

settimanaGRID <- unique(incidenti$settimana)
comuniGRID <- comuni$NOME

#creo griglia di previsione con target per confronto con previsione in training 
griglia <- expand_grid(NOME = comuniGRID, settimana = settimanaGRID)
incidentiweek <- griglia %>% left_join(incidentiweek, by = c("NOME", "settimana")) %>% mutate(n_incidenti = replace_na(n_incidenti, 0),
                                                                                              target = if_else(n_incidenti > 0, 1, 0))

#Ho bisogno di un punto rappresentativo per il meteo comunale della settimana
#Provato centro del comune ma vengono fuori troppe chiamate
comuniCENTRO <- st_centroid(comuni) %>% mutate(lon = st_coordinates(.)[,1], lat = st_coordinates(.)[,2]) %>% st_drop_geometry() %>% select(NOME, lat, lon)

#creo tot punti a cui fanno riferimento i vari comuni (da ~250 a 25)
comuniAGG <- kmeans(comuniCENTRO %>% select(lat, lon), centers = 25)

comuniCENTRO <- comuniCENTRO %>% mutate(zona = comuniAGG$cluster)

zonaCENTRO <- as.data.frame(comuniAGG$centers) %>% mutate(zona = row_number())

#Controllo 3 giorni a settimana per ogni punto, valutare se aumentare a 5/7
incidentiweek <- incidentiweek %>% mutate(weeknum = as.numeric(substr(settimana, 6, 7)),
                                          lun = as.Date(paste(data = as.numeric(substr(settimana, 1, 4)), weeknum, 1), format = "%Y %U %u"),
                                          mer = lun + 2,
                                          ven = lun + 4)

#distinct da righe uniche del dataset
meteoCOMUNI <- incidentiweek %>% distinct(NOME, settimana, lun, mer, ven) %>% left_join(comuniCENTRO, by = "NOME")

#METEO SINGOLI COMUNI, serve per confronto training

getmeteoZONE <- function(data) {
  res <- GET(
    "https://archive-api.open-meteo.com/v1/archive",
      query = list(
      latitude = paste(zonaCENTRO$lat, collapse = ","),
      longitude = paste(zonaCENTRO$lon, collapse = ","),
      start_date = data,
      end_date = data,
      daily = "temperature_2m_mean,precipitation_sum,windspeed_10m_max",
      timezone = "Europe/Rome"
    ))
  if (status_code(res) != 200) {
    return(NULL)
  }
  
datiM2 <- fromJSON(content(res, as = "text", encoding = "UTF-8"))
  
tibble(
  zona = zonaCENTRO$zona,
  data_iso = data,
  tempMEAN = datiM2$daily$temperature_2m_mean,
  precipMEAN = datiM2$daily$precipitation_sum,
  ventoMEAN = datiM2$daily$windspeed_10m_max)
}


dateUNICHE <- incidentiweek %>% select(lun, mer, ven) %>% 
                                pivot_longer(everything(), values_to = "data_iso") %>%
                                distinct(data_iso) %>%
                                pull(data_iso)



#call API METEO
meteoZONE <- map_dfr(dateUNICHE, function(data) {
  Sys.sleep(0.1)
  getmeteoZONE(data)
})








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

#valutare se tenere GETTERRENI da openstreetmap per altitudine (guarda legacy)
#analizzare risultati snap +/- di 30? snap corretto?


