# Data preparation

# Set up ====
rm(list=ls())

# ..... Libraries ====
library(tidyverse)
library(ggbeeswarm)
library(snakecase)

library(parallel)
library(parallelsugar)

mclapply <- switch( Sys.info()[['sysname']],
                    Windows = {mclapply_socket},
                    Linux   = {mclapply},
                    Darwin  = {mclapply})


# ..... Flags ====

save_output_data = T

p.scatter = T
p.boxplot = T
p.shaps = T
p.pu = F
p.pu_kenya = F
p.pu_error_by_country = F
p.xgb_vs_gam = F
p.xgb_vs_gam_scatter = F
p.pu_preds_gam = F
p.pu_preds_gam_kenya = F
p.xgb_vs_gam_map = F


# ..... Variants table ====

response_var <- c('vci', 'ndvi')

filter_function <- list(
  
  wio = function(data) { 
    data %>% return() 
    },
                        
  kenya = function(data) { 
    data %>% 
      filter(country == "kenya") %>% 
      return() 
    }
  )

out_dir_extra <- c('wio', 'kenya')

variant_table <- tibble(out_dir_extra = out_dir_extra,
                        filter_function = filter_function
                        ) %>%
  expand_grid(response_var)


# ..... Directories and files ====
in_dir.data <- "C:/Users/MQ43846173/Dropbox/Lenfest_kenya/data/"

in_dir.mangrove <- 'habitat_cover/mangrove/maina_et.al_2020_mangrove_model/'
in_file.mangrove <- 'all.data.csv'

in_file.mangrove_raw <- 'WIO_mangrove_cover_plus_variables_raw.csv'


out_dir.start <- "output/mangrove/"


# ..... Constants ====


id_vars.character <- c("id",
                       "region",
                       "country")

id_vars.numeric <- c("pu_lon",
                     "pu_lat"
                     )

id_vars <- c(id_vars.character, id_vars.numeric)


replicate_unit_var <- "UNKNOWN_pu_id"

features.numeric <- c(
  
 
  # habitat variables
  #"coastal_type",
  
  # Physical variables
  "elevation",
  "slope",
  "tide_cm",  # Average tide height/ Tidal range?
  "UNKNOWN_average_mean_sea_level",
  
  
  # Anthropogenic variables
  "global_erosion_borrelli",
  "human_gravity",
  "land_dev_intensity",
  "market_gravity_mean",
  "market_gravity_sd",
  "UNKNOWN_land_dev_intesnity_ersome",
  "UNKNOWN_land_dev_intesnity_ersost",
  "travel_time_to_nearest_market",
  
  
  # Threat variables
  "sea_level_anomaly",
  "warm_days_90pct",
  "consecutive_dry_days",
  
  # Other
  "UNKNOWN_detided",
  "bcs.msl",
  "bcs.msl"
  
  
)

features.one_hot <- c('coastal_type')

features <- c(features.numeric, features.one_hot)


forecasting_vars <- c("sea_level_rise_rate",
                      "consecutive_dry_days_rcp85",
                      "mean_sea_level_rcp26_2050",
                      "mean_sea_level_rcp45_2050",
                      "mean_sea_level_rcp60_2050",
                      "mean_sea_level_rcp85_2050",
                      "mean_sea_level_rcp26_trend",
                      "mean_sea_level_rcp45_trend",
                      "mean_sea_level_rcp85_trend",
                      "wcs.trend",
                      "bcs.trend",
                      "warm_days_90pct_rcp85")
# Features to add 

## ACA Geomorphology - Presence/Absence of each class
## 


response_type <- 'regression'
n_runs <- 50

train_test_split_val <- 0.1
train_watchlist_split_val <- 0.2



# Main ==== 
# Variant loop

lapply(1:nrow(variant_table), function(x) {
  
  # ..... Get variant values ====
  
  out_dir_extra <- variant_table$out_dir_extra[[x]]
  filter_function <- variant_table$filter_function[[x]]
  response_var <- variant_table$response_var[[x]]
  
  response_var_out <- to_snake_case(response_var)
  
  
  # ..... Create directories
  out_dir <- paste0(out_dir.start,response_var_out,'/',out_dir_extra,'/')
  
  if(!dir.exists(out_dir)) { dir.create(out_dir, recursive = T,) }
  
  
  # ..... Secondary constants
  actual_var <- paste0(response_var,'.actual')
  predict_var <- paste0(response_var,'.predicted')
  
  
  # ..... Dataset clean up ====
  
  data.mangrove <- read_csv(paste0(in_dir.data,in_dir.mangrove,in_file.mangrove))
  
  #data.mangrove_raw <- read_csv(paste0(in_dir.data,in_dir.mangrove,in_file.mangrove_raw))
  
  
  data.mangrove <- data.mangrove %>%
    
    dplyr::select(-`...1`) %>%
    
    # Renaming variables to more descriptive ones
    rename(UNKNOWN_pu_id = X,
           pu_lon = x,
           pu_lat = y,
           vci = vci,
           elevation = elevation,
           global_erosion_borrelli = erosion,
           human_gravity = gravity,
           land_dev_intensity = ldi,
           sea_level_anomaly = sla,
           slope = slope,
           tide_cm = tidecm,  # Average tide height/ Tidal range?
           warm_days_90pct = tx90,
           consecutive_dry_days = ccd,
           UNKNOWN_detided = detided,
           id = id,
           region = layer,
           country = County,
           market_gravity_mean = GoMmean,
           market_gravity_sd = GoMstdev,
           UNKNOWN_land_dev_intesnity_ersome = ldi.ersome,
           UNKNOWN_land_dev_intesnity_ersost = ldi.ersost,
           sea_level_rise_rate = slr,
           ndvi = ndvi,
           travel_time_to_nearest_market = travel.time,
           UNKNOWN_average_mean_sea_level = avmsl,
           consecutive_dry_days_rcp85 = cdd.rcp85,
           mean_sea_level_rcp26_2050 = msl26.2050,
           mean_sea_level_rcp45_2050 = msl45.2050,
           mean_sea_level_rcp60_2050 = msl60.2050,
           mean_sea_level_rcp85_2050 = msl85.2050,
           mean_sea_level_rcp26_trend = msl.26.trend,
           mean_sea_level_rcp45_trend = msl.45.trend,
           mean_sea_level_rcp85_trend = msl.85.trend,
           wcs.trend = wcs.trend,
           bcs.trend = bcs.trend,
           warm_days_90pct_rcp85 = tx9085,
           bcs.msl = bcs.msl,
           wcs.msl = wcs.msl,
           coastal_type = formation
           ) %>%
    
    mutate(across(any_of(c(replicate_unit_var, id_vars.character, features.one_hot)), as.character)) %>%
    
    mutate(across(any_of(c(replicate_unit_var, id_vars.character, features.one_hot)), to_snake_case)) %>%
    
    
    mutate(pu_lat = as.numeric(pu_lat),
           pu_lon = as.numeric(pu_lon)) %>%

    return()
  


  # ..... Get model data ====
  
  model_data <- data.mangrove
  
  model_data <- model_data %>%
    filter_function() %>%
    return()
  
  model_data <- model_data[complete.cases(model_data), ]
  
  
  # ..... Run models ====
  
  source('habitat_modelling_pipeline.r', local = T)
  
  
  # ..... Generate plots ====  
  source('model_plotting.r', local = T)
  

})
