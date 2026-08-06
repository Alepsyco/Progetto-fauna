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
    return(tibble(lat = NA, long = NA, data_iso = NA, precipitazioni = NA))
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
    vento = datiM$hourly$windspeed_10m[indiceORA]
  )
}  

#FUNZIONE CHIAMATA API SU TUTTO DATASET TOKEN LIMITATIN E LENTO ----NON ESEGUIRE A CASO----
  
meteo <- pmap_dfr(incidenti %>% st_drop_geometry() %>% select(lat, long, data_iso, orario),     #dropgeometry se no si rompe
                  function(lat, long, data_iso, orario) {
                    Sys.sleep(0.1) #pausa per non rompere api
                    getmeteo(lat, long, data_iso, orario)
  }
)

incidenti <- incidenti %>% bind_cols(meteo %>% select(temp, precipitazioni, vento))


#-------------------------------------------------

#GETTERRENI da openstreetmaps

terreni <- st_read(
  "./06_Friuli-Venezia_Giulia-2026-07-31T05Z.osm.pbf", query = "SELECT * FROM multipolygons WHERE natural IN ('wood','scrub','tree_row') OR 
  landuse = 'forest' OR
  landuse IN ('cemetery','religious','education','recreation_ground','village_green','allotments','greenfield','farmland','farmyard','orchard','vineyard','meadow','animal_keeping','aquaculture','plant_nursery')"
)


terreni <- terreni %>% select(natural, landuse, building, geometry = "_ogr_geometry_") #limita colonne interessate
terreni <- terreni %>% st_make_valid() #ripara terreni non chiusi

sum(!st_is_valid(terreni)) #controlla che tutti i terreni siano riparati

#terreniUNION <- terreni %>% group_by(natural) %>% summarise(geometry = st_union(geometry))


###################################################################
###################################################################

#TODO

#sOSTITUIRE DATASET TERRENI
#aggiustare dati e decidere come trattare incidenti
#analizzare risultati snap +/- di 30? snap corretto?


