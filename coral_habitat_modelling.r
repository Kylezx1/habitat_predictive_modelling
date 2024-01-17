# Data preparation

# Set up ====
rm(list=ls())

# ..... Libraries ====
library(tidyverse)
library(ggbeeswarm)
library(snakecase)

source("machine_learning_functions.r")

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

p.pu = T
p.pu_kenya = T
p.pu_error_by_country = T

p.xgb_vs_gam = F
p.xgb_vs_gam_scatter = F
p.pu_preds_gam = F
p.pu_preds_gam_kenya = F
p.xgb_vs_gam_map = F



# ..... Variants table ====

response_var <- c('number_of_genera', 'coral_cover')


filter_function <- list(
  
  wio = function(data) { 
    data %>% 
      filter(number_of_replicates == 1) %>% 
      return() 
    },
  
  kenya = function(data) { 
    data %>% 
      filter(country == "kenya") %>% 
      filter(number_of_replicates == 1) %>% 
      return() 
    }
  )

filter_function_pu <- list(
  wio = function(data) { 
    data %>% 
      return() 
    },
  
  kenya = function(data) { 
    data %>% 
      filter(country == "kenya") %>% 
      return() 
    }
  )


out_dir_extra <- c('1_rep_wio', '1_rep_kenya')

variant_table <- tibble(out_dir_extra = out_dir_extra,
                        filter_function = filter_function,
                        filter_function_pu = filter_function_pu) %>%
  expand_grid(response_var)


# ..... Directories and files ====
in_dir.data <- "C:/Users/MQ43846173/Dropbox/Lenfest_kenya/data/"

in_dir.coral <- 'habitat_cover/coral/field_observations/mcclanahan_2023/'
in_file.coral <- 'WIO_coral_cover_plus_variables_aggregated.csv'

in_file.coral_raw <- 'WIO_coral_cover_plus_variables_raw.csv'


out_dir <- "output/coral/"

# ..... Constants ====


id_vars.character <- c("country",
                       "ecoregion",
                       'observer')

id_vars.numeric <- c("pu_lat",
                     "pu_lon",
                     "min_yr",
                     "max_yr",
                     "number_of_replicates")

id_vars <- c(id_vars.character, id_vars.numeric)

                       

replicate_unit_var <- "pu_id"

features.numeric <- c(
  
  # Site variables
  # "habitat",               # Reef habitat type
  "depth",
  
  
  # Threats
  
  # The metadata descriptions for these SST metrics
  # here suggests that the SST data have not been
  # subset by year. Potential problem.
  "sst_rate_of_rise",         # slope of the Kendall trend test on mean annual SST values for the 1985 to 2020 period
  "sst_kurtosis",          # from 1985-2020
  "sst_median",            # from 1985-2020       
  "sst_bimodality",        # from 1985-2020
  "sst_skewness",          # from 1985-2020
  "cumulative_dhw",                # from 1985-2020
  
  "climate_stress_model",    # Multivariate climate stress (years?)
  
  # Environmental factors
  
  # We don't know what years/year ranges these data are from.
  "par_max",                
  "calcite",               
  "dissolved_oxygen",            
  "salinity_mean",         # What dataset? 
  "chlor_a_median",         
  "ph",
  "diffuse_attenuation",      # Diffuse water penetration coefficient
  
  "mean_wave_energy",          # Mean wave energy. What dataset? 
  "current_velocity_median",      # From what dataset?
  
  "andrello_reef_value",   # What is the reef value?
  "andrello_nutrients",
  "andrello_sediments",
  
  
  # Human variables
  
  # "Management",            # Fisheries management level
  
  "mean_net_primary_productivity",              # Primary productivity, from where?
  "time_to_market",         # Travel time to market
  "time_to_population",            # Travel time to nearest population
  
  "gravity_to_nearest_population",               # Gravity to nearest population
  "gravity_to_nearest_market",               # Gravity to nearest city or market
  
  
  # Connectivity
  
  "netflow",
  "indegree",
  "outdegree",
  "retention"
  
)

features.one_hot <- c("management",
                      "habitat")

features <- c(features.numeric, features.one_hot)

# Features to add 

## ACA Geomorphology - Presence/Absence of each class
## 


response_type <- 'regression'
n_runs <- 50

train_test_split_val <- 0.1
train_watchlist_split_val <- 0.2


global_objective <- "reg:squarederror"
global_objective <- "count:poisson"


# Main ==== 
# Variant loop

lapply(1:nrow(variant_table), function(x) {
    
  # ..... Get variant values ====
  
  out_dir_extra <- variant_table$out_dir_extra[[x]]
  filter_function <- variant_table$filter_function[[x]]
  filter_function_pu <- variant_table$filter_function_pu[[x]]
  response_var <- variant_table$response_var[[x]]
  
  response_var_out <- to_snake_case(response_var)
  
  # ..... Create directories ====
  out_dir <- paste0(out_dir,response_var_out,'/',out_dir_extra,'/')
  
  if(!dir.exists(out_dir)) { dir.create(out_dir, recursive = T,) }
  
  
  # ..... Secondary constants ====
  actual_var <- paste0(response_var,'.actual')
  predict_var <- paste0(response_var,'.predicted')
  
  response_var_out <- to_snake_case(response_var)
  
  # ..... Dataset clean up ====
  # .......... Model dataset ====
  
  data.coral <- read_csv(paste0(in_dir.data,in_dir.coral,in_file.coral))
  
  names(data.coral) <- names(data.coral) %>% to_snake_case()
  
  data.coral <- data.coral %>%
    
    # pu_lats and longs are wrong
    rename(pu_lon = pu_lat,
           pu_lat = pu_lon) %>%
    
    # Renaming
    rename(sst_rate_of_rise = sst_rateof_rise,
           par_max = pa_rmax,
           dissolved_oxygen = dis_oxygen,
           mean_wave_energy = mean_wave_nrj,
           time_to_market = tt_market_hrs,
           time_to_population = tt_pop_hrs,
           cumulative_dhw = cum_dhw,
           current_velocity_median = current_vel_mean,
           diffuse_attenuation = diff_attenuation,
           gravity_to_nearest_population = grav_np,
           gravity_to_nearest_market = grav_nc,
           mean_net_primary_productivity = mean_npp,
           number_of_replicates = no_replicates,
           coral_cover = coralcover) %>%
    
    mutate(across(any_of(c(replicate_unit_var, id_vars.character, features.one_hot)), as.character)) %>%

    mutate(across(any_of(c(replicate_unit_var, id_vars.character, features.one_hot)), to_snake_case)) %>%
    
    mutate(coral_cover_prop = coral_cover/100) %>% 
    mutate(pu_lat = as.numeric(pu_lat),
           pu_lon = as.numeric(pu_lon)) %>%
    
    return()
  
  
  
  
  
  # .......... Planning unit dataset ====
  
  data.coral_raw <- read_csv(paste0(in_dir.data,in_dir.coral,in_file.coral_raw))
  
  names(data.coral_raw) <- names(data.coral_raw) %>% to_snake_case()
  
  data.coral_raw <- data.coral_raw %>%
    
    select(-1) %>%
    
    # pu_lats and longs are wrong
    rename(pu_lon = lon,
           pu_lat = lat) %>%
    
    # Renaming
    rename(sst_rate_of_rise = sst_rateof_rise,
           par_max = pa_rmax,
           dissolved_oxygen = dis_oxygen,
           mean_wave_energy = mean_wave_nrj,
           time_to_market = tt_market_hrs,
           time_to_population = tt_pop_hrs,
           cumulative_dhw = cum_dhw,
           current_velocity_median = current_vel_mean,
           diffuse_attenuation = diff_attenuation,
           gravity_to_nearest_population = grav_np,
           gravity_to_nearest_market = grav_nc,
           mean_net_primary_productivity = mean_npp,
           # number_of_replicates = no_replicates,
           coral_cover = coralcover) %>%
    
    mutate(across(any_of(c(replicate_unit_var, id_vars.character, features.one_hot)), as.character)) %>%
    
    mutate(across(any_of(c(replicate_unit_var, id_vars.character, features.one_hot)), to_snake_case)) %>%
    
    mutate(coral_cover_prop = coral_cover/100) %>% 
    mutate(pu_lat = as.numeric(pu_lat),
           pu_lon = as.numeric(pu_lon)) %>%
    
    mutate(number_of_genera = 0) %>%  # Needed for pu predictions
  
    return()
  
  # ..... Get model data ====
  
  model_data <- data.coral
  
  model_data <- model_data %>%
    filter_function() %>%
    return()
  
  model_data <- model_data[complete.cases(model_data), ]
  
  
  pu_data <- data.coral_raw %>%
    filter_function_pu() %>%
    return()
  
  
  # ..... Run models ====
  
  source('habitat_modelling_pipeline.r', local = T)
  
  
  # ..... Generate plots ====  
  source('model_plotting.r', local = T)
  
  
})
