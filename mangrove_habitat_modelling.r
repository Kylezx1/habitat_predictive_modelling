# Data preparation

# Set up ====
rm(list=ls())

# Libraries
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


# Directories and files
in_dir.data <- "C:/Users/MQ43846173/Dropbox/Lenfest_kenya/data/"

in_dir.mangrove <- 'habitat_cover/mangrove/maina_et.al_2020_mangrove_model/'
in_file.mangrove <- 'all.data.csv'

in_file.mangrove_raw <- 'WIO_mangrove_cover_plus_variables_raw.csv'


out_dir <- "output/mangrove/"
out_dir_extra <- 'WIO/'

# Filtering
data_filter <- function(data) {
  data %>%
    # filter(no_replicates == 1) %>%
    # filter(Country == "Tanzania") %>%
    # filter(country == "Kenya") %>%
    return()
}


data_filter.pu <- function(data) {
  data %>%
    # filter(Country == "Tanzania") %>%
    # filter(Country == "Kenya") %>%
    return()
}


# Constants
response_var <- 'vci'
response_var <- "vci"
response_var_out <- to_snake_case(response_var)


id_vars <- c("PU_lon",
             "PU_lat",
             "id",
             "region",
             "country")

replicate_unit_var <- "UNKNOWN_pu_id"

features <- c(
  
 
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


features.one_hot <- c('coastal_type')

# Features to add 

## ACA Geomorphology - Presence/Absence of each class
## 


response_type <- 'regression'
n_runs <- 100

train_test_split_val <- 0.1
train_watchlist_split_val <- 0.2



# Main ==== 
# ..... Create directories
out_dir <- paste0(out_dir,response_var_out,'/',out_dir_extra)

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
         PU_lon = x,
         PU_lat = y,
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

  mutate(across(any_of(c(replicate_unit_var, id_vars)), as.character)) %>%
  mutate(PU_lat = as.numeric(PU_lat),
         PU_lon = as.numeric(PU_lon)) %>%
  
  # TODO - Finish this
  # mutate(across(any_of(features.one_hot)), caret::dummyVars) %>%
  
  return()


# data.mangrove_raw <- data.mangrove_raw %>%
#   rename(PU_lat = Lat,
#          PU_lon = Lon) %>%
#   
#   mutate(across(any_of(c(replicate_unit_var, id_vars)), as.character)) %>%
#   mutate(Coralcover.prop = Coralcover/100) %>%
#   mutate(PU_lat = as.numeric(PU_lat),
#          PU_lon = as.numeric(PU_lon)) %>%
#   
#   mutate(Number_of_genera = 0)  # Needed for PU predictions



# ..... Get model data ====

model_data <- data.mangrove

model_data <- model_data %>%
  data_filter() %>%
  return()


model_data <- model_data[complete.cases(model_data), ]


# pu_data <- data.mangrove_raw %>%
#   data_filter.pu() %>%
#   return()


# ..... Run models ====

source('habitat_modelling_pipeline.r')


# # ..... Single model run ====
# 
# xgb_model <- 
#   xgboost_model_training_wrapper(input_data = model_data,
#                                  response_var = response_var,
#                                  replicate_unit_var = replicate_unit_var,
#                                  id_vars = id_vars,
#                                  features = features,
#                                  response_type = response_type,
#                                  train_test_split_val = train_test_split_val,
#                                  train_watchlist_split_val = train_watchlist_split_val,
#                                  
#                                  xgb_booster = 'gbtree',
#                                  global_objective = "reg:squarederror",  # Use for regression
#                                  global_eval_metric = 'rmse',
#                                  
#                                  tree_max_depth = 12,   # Adjust as needed
#                                  tree_eta = 0.01,       # Learning rate
#                                  tree_subsample = 0.5,
#                                  tree_lambda = 0.5,
#                                  tree_alpha = 0.5,
#                                  tree_num_parallel_tree = 3,
#                                  
#                                  
#                                  nthread = 12,
#                                  
#                                  # xgb_booster = 'gbtree',
#                                  # global_objective = custom_obj_beta_regression,
#                                  # global_eval_metric = 'logloss',
#                                  # maximize = TRUE,
#                                  
#                                  linear_lambda_bias = NULL,
#                                  
#                                  training_rounds = 20000,
#                                  early_stopping_rounds = 50,
#                                  print_progress = T)
# 
# 
# xgb_model.predictions <- predict(object = xgb_model$xgboost_model,
#                                  newdata = xgb_model$model_matrices.xgb$test,
#                                  data = xgb_model$model_matrices.xgb$train)
# 
# 
# dnorm(xgb_model.predictions)
# 
# xgb_model.predictions <- tibble(!!predict_var := xgb_model.predictions) %>%
#   bind_cols(xgb_model$xgb_data_list$test[,c(response_var, replicate_unit_var, id_vars)]) %>%
#   rename(!!actual_var := response_var)
# 
# xgb_model.predictions <- xgb_model.predictions %>%
#   mutate(error = get(predict_var) - get(actual_var),
#          error.abs = abs(error),
#          error.pc = (error / get(actual_var)) * 100,
#          error.abs.pc = abs(error.pc))
# 
# 
# # ..... Multiple runs ====
# # .......... Run models ====
# 
# xgb_models <- mclapply(1:n_runs, function(x) {
#   
#   print(paste0("running model ",x," of ", n_runs))
#   
#   xgb_model <- 
#     xgboost_model_training_wrapper(input_data = model_data,
#                                    response_var = response_var,
#                                    replicate_unit_var = replicate_unit_var,
#                                    id_vars = id_vars,
#                                    features = features,
#                                    response_type = response_type,
#                                    train_test_split_val = train_test_split_val,
#                                    train_watchlist_split_val = train_watchlist_split_val,
#                                    
#                                    xgb_booster = 'gbtree',
#                                    global_objective = "reg:squarederror",  # Use for regression
#                                    global_eval_metric = 'rmse',
#                                    global_nthread = 12,
#                                    
#                                    
#                                    tree_max_depth = 12,   # Adjust as needed
#                                    tree_eta = 0.01,       # Learning rate
#                                    tree_subsample = 0.5,
#                                    tree_lambda = 0.5,
#                                    tree_alpha = 0.5,
#                                    tree_num_parallel_tree = 3,
#                                    
#                                    
#                                    
#                                    # xgb_booster = 'gbtree',
#                                    # global_objective = custom_obj_beta_regression,
#                                    # global_eval_metric = 'logloss',
#                                    # maximize = TRUE,
#                                    
#                                    
#                                    training_rounds = 20000,
#                                    early_stopping_rounds = 50,
#                                    print_progress = T
#     )
#   
#   
#   predictions <- predict(object = xgb_model$xgboost_model,
#                          newdata = xgb_model$model_matrices.xgb$test,
#                          data = xgb_model$model_matrices.xgb$train)
#   
#   predictions <- tibble(!!predict_var := predictions) %>%
#     bind_cols(xgb_model$xgb_data_list$test[,c(response_var, replicate_unit_var, id_vars)]) %>%
#     rename(!!actual_var := response_var)
#   
#   mean_actual <-  mean(predictions[[actual_var]])
#   
#   predictions <- predictions %>%
#     mutate(run = x,
#            mean_actual = mean_actual,
#            error = get(predict_var) - get(actual_var),
#            error.abs = abs(error),
#            error.pc = (error / get(actual_var)) * 100,
#            error.abs.pc = abs(error.pc),
#            error_mean = get(predict_var) - mean_actual,
#            error_mean.abs = abs(error_mean)) %>%
#     relocate(run, all_of(id_vars))
#   
#   xgb_model[['predictions']] <- predictions
#   
#   shaps_x <- get_shap_values.xgboost(trained_xgb_model_object = xgb_model,
#                                      response_var = response_var,
#                                      predinteraction = F,
#                                      problem_class = 'regression')
#   shaps_x <- shaps_x %>% mutate(run = x)
#   
#   
#   xgb_model[['shap_values']] <- shaps_x
#   
#   xgb_model
#   
# }, mc.cores = 4)
# 
# 
# 
# # .......... Model performance ====
# 
# xgb_models.predictions <- lapply(1:n_runs, function(x) {
#   xgb_models[[x]]$predictions
# }) %>% bind_rows() 
# 
# 
# model_runs.performance.summary <- xgb_models.predictions %>%
#   #pivot_longer(cols = c('accuracy', 'confidence'), names_to = 'metric') %>%
#   group_by(across(c('run', 'Country'))) %>%
#   summarise(error.mean = mean(error),
#             error.sd = sd(error),
#             error_abs.mean = mean(error.abs),
#             mean_only_error = mean(error_mean),
#             mean_only_error.abs = mean(error_mean.abs)
#             # confidence.mean = mean(confidence),
#             # confidence.sd = sd(confidence)
#   )
# 
# 
# model_runs.performance.summary.mean_only_models <- model_runs.performance.summary %>%
#   group_by(Country) %>%
#   summarise(mean_only_error = mean(mean_only_error),
#             mean_only_error.abs = mean(mean_only_error.abs),
#             median_only_error = median(mean_only_error),
#             median_only_error.abs = median(mean_only_error.abs))
# 
# 
# model_runs.performance.country_error <- model_runs.performance.summary %>%
#   group_by(Country) %>%
#   summarise(model_error_rate.country = mean(mean_only_error.abs))
# 
# 
# # .......... SHAP values ====
# 
# shap_values <- lapply(1:4, function(x) {
#   
#   out <- xgb_models[[x]]$shap_values
# 
#   out
#   
# }) %>% bind_rows() 
# 
# 
# shap_values.long <- shap_values %>%
#   pivot_longer(cols = features,
#                names_to = 'feature',
#                values_to = 'shap') %>%
#   mutate(shap_abs = abs(shap)) %>%
#   arrange(-shap_abs)
# 
# 
# 
# # .......... Predict to PUs ====
# 
# 
# pu_data.xgb_matrix <- convert_data_to_xgb_matrix(xgb_data_list = list(pu_data),
#                                                  features = features,
#                                                  response_var = response_var,
#                                                  response_type = response_type) %>%
#   .[[1]]
# 
# 
# pu_predictions <- lapply(1:n_runs, function(x) {
#   
#   model_x <- xgb_models[[x]]$xgboost_model
#   data_x <- xgb_models[[x]]$xgb_data_list$train
#   preds_x <- predict(object = model_x, 
#                      data = data_x,
#                      newdata = pu_data.xgb_matrix)
#   
#   out <- tibble(!!predict_var := preds_x,
#                 model_run = x) %>%
#     bind_cols(pu_data)
#   
#   out
#   
# }) %>% bind_rows()
# 
# 
# pu_preds.aggregated <- pu_predictions %>%
#   group_by(across(any_of(c(id_vars)))) %>%
#   summarise(!!paste0(predict_var,'.mean') := mean(get(predict_var)),
#             !!paste0(predict_var,'.sd') := sd(get(predict_var)),
#             !!paste0(predict_var,'.GAM') := mean(Coralcover.prop),
#             !!paste0(predict_var,'.GAM_sd_check') := sd(Coralcover.prop)
#   ) %>%
#   ungroup() %>%
#   left_join(model_runs.performance.country_error) %>%
#   mutate(xgb_vs_gam_diff = get(paste0(predict_var,'.mean')) - get(paste0(predict_var,'.GAM'))) %>%
#   mutate(PU_lat = as.numeric(PU_lat),
#          PU_lon = as.numeric(PU_lon))
# 
# 
# pu_preds_pivot_vars <- c(paste0(predict_var,'.mean'),
#                          paste0(predict_var,'.sd'),
#                          'model_error_rate.country')
# 
# pu_preds.aggregated



# ..... Results and plots ====
# .......... Model performance summary - scatter ====

plot <- ggplot(data = xgb_models.predictions,
               aes(x = get(predict_var), y = get(actual_var))) +
  
  geom_abline(slope = 1, intercept = 0, size = 2, colour = 'red') +
  
  geom_point() +
  
  theme_minimal(base_size = 22) + 
  theme(#text = element_text(colour = '#00632e'), 
    #axis.text = element_text(colour = '#00632e'),
    #axis.ticks = element_line(colour = '#00632e', size = 2),
    #axis.line = element_line(colour = '#00632e', size = 2),
    #strip.text = element_text(colour = '#00632e'),
    
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    
    strip.background = element_blank(),
    #panel.border = element_rect(colour = '#00632e', fill = NA, size = 2),
    panel.background = element_rect(fill = "white", colour = '#00632e'),
    plot.background =  element_rect(fill = "white", colour = '#00632e', size = 2),
    #legend.background = element_rect(fill = "white", colour = NA),
    
    legend.position="bottom",
  ) 

plot

ggsave(filename = paste0(out_dir,'model_performance_tree_booster',response_var_out,'_scatter.png'),
       plot = plot,
       width = 1200,
       height = 1080,
       scale = 3,
       units = 'px',
       overwrite = F)


# .......... Model performance summary - boxplot ====

plot <- ggplot(data = model_runs.performance.summary,
               aes(x = '', y =  error_abs.mean, colour = country)) +
  
  geom_hline(data = model_runs.performance.summary.mean_only_models,
             aes(colour = country,
                 yintercept = mean_only_error.abs),
             size = 2, alpha = 0.6) +
  
  geom_boxplot(size = 1, show.legend = T) +
  # geom_beeswarm(size = 3, alpha = 0.2, method = 'hex',
  #               show.legend = T) +
  
  facet_wrap(~country, scales = 'free_y') +
  
  
  # scale_fill_viridis_d(begin = 0.2, end = 0.8, option = 'D') +
  # scale_colour_viridis_d(begin = 0.2, end = 0.8, option = 'D') +
  
  scale_colour_manual(values = interleave_colors(11, 2, 0.1, 0.9)) +
  
  labs(colour = 'country',
       y = 'Error (absolute)',
       x = '') +
  
  theme_minimal(base_size = 20) + 
  theme(#text = element_text(colour = '#00632e'), 
    #axis.text = element_text(colour = '#00632e'),
    #axis.ticks = element_line(colour = '#00632e', size = 2),
    #axis.line = element_line(colour = '#00632e', size = 2),
    #strip.text = element_text(colour = '#00632e'),
    
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    
    strip.background = element_blank(),
    #panel.border = element_rect(colour = '#00632e', fill = NA, size = 2),
    panel.background = element_rect(fill = "white", colour = '#00632e'),
    plot.background =  element_rect(fill = "white", colour = '#00632e', size = 2),
    #legend.background = element_rect(fill = "white", colour = NA),
    
    legend.position="bottom",
  ) 

plot

ggsave(filename = paste0(out_dir,'model_performance_tree_booster_',response_var_out,'_boxplot.png'),
       plot = plot,
       width = 1200,
       height = 1080,
       scale = 3,
       units = 'px',
       overwrite = F)


# ..........  Feature importance ====

shap_plot <- ggplot(data = shap_values.long %>% filter(data_type == 'train'),
                    aes(y = shap_abs, x = reorder(feature, -shap_abs))) +
  
  geom_hline(yintercept = 0) +
  
  geom_bar(fill = '#008822', alpha = 0.6, size = 1.5,
           stat = 'summary', fun = 'sum') +
  
  scale_y_continuous(expand = c(0, 0)) + 
  
  labs(x = 'feature') +
  
  theme_bw() +
  theme(# axis.text.x = element_blank(),
    # axis.ticks.x = element_blank(),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
    
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid = element_blank())

shap_plot

ggsave(filename = paste0(out_dir,'feature_importance_shap_values_',response_var_out,'.png'),
       plot = shap_plot,
       width = 1900,
       height = 1080,
       scale = 1,
       units = 'px',
       overwrite = F)


# ..... Savin' data ====

write_csv(x = model_runs.performance.country_error,
          file = paste0(out_dir,'model_performance_by_country.csv'))


write_csv(x = model_runs.performance.summary,
          file = paste0(out_dir,'model_performance_summary.csv'))


write_csv(x = xgb_models.predictions,
          file = paste0(out_dir,'model_predictions_raw.csv'))


# write_csv(x = shap_values,
#           file = paste0(out_dir,'shap_values.csv'))


stop()



# .......... Map

prediction_summary <- xgb_models.predictions %>%
  group_by(across(id_vars)) %>%
  summarise(mean_prediction = mean(get(predict_var)),
            sd_prediction = sd(get(predict_var)),
            mean_abs_error = mean(error.abs),
            sd_abs_error = sd(error.abs)) %>%
  pivot_longer(cols = c('mean_prediction', 
                        'sd_prediction',
                        'mean_abs_error',
                        'sd_abs_error'),
               names_to = 'variable',
               values_to = 'value')

plot <- ggplot(data = prediction_summary %>% filter(variable == 'mean_abs_error'),
               aes(x = PU_lon, y = PU_lat, colour = value)) +
  
  geom_point(alpha = 0.3) +
  
  facet_wrap(~variable) +
  
  scale_color_viridis_c() +
  
  theme_bw()
plot


# ..........  PU predictions ====

pu_preds.plot <- ggplot(data = pu_preds.aggregated %>% filter(Country == 'Kenya'),
                        mapping = aes(x = PU_lon, 
                                      y = PU_lat,
                                      colour = get(paste0(predict_var,'.mean')))) +
  
  geom_point(size = 2, alpha = 0.6) +
  
  coord_equal() +
  
  labs(colour = paste0('Mean predicted\n',response_var_out,'\nXGBoost')) +
  
  theme_bw() +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())

pu_preds.plot


ggsave(filename = paste0(out_dir,'planning_units_predictions_kenya_',response_var_out,'.png'),
       plot = pu_preds.plot,
       width = 1200,
       height = 1080,
       scale = 3,
       units = 'px')


# .......... PU predictions - Kenya ====

pu_preds.plot <- ggplot(data = pu_preds.aggregated %>% filter(Country == 'Kenya'),
                        mapping = aes(x = PU_lon, 
                                      y = PU_lat,
                                      colour = get(paste0(predict_var,'.sd')))) +
  
  geom_point(size = 2, alpha = 0.6) +
  
  coord_equal() +
  
  scale_color_gradient(low = "#CCCC00",
                       high = "#CC0000") +
  
  labs(colour = 'Model ensemble\nagreement\nXGBoost') +
  
  theme_bw() +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())

pu_preds.plot


ggsave(filename = paste0(out_dir,'planning_units_prediction_uncertainty_kenya_',response_var_out,'.png'),
       plot = pu_preds.plot,
       width = 1200,
       height = 1080,
       scale = 3,
       units = 'px')


# .......... Model error by country ====

pu_preds.plot <- ggplot(data = pu_preds.aggregated,
                        mapping = aes(x = PU_lon, 
                                      y = PU_lat,
                                      colour = model_error_rate.country)) +
  
  geom_point(size = 1, alpha = 0.6) +
  
  coord_equal() +
  
  scale_color_gradient(low = "#CCCC00",
                       high = "#CC0000") +
  
  labs(colour = 'Model error rate\nby country') +
  
  theme_bw() +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())

pu_preds.plot


ggsave(filename = paste0(out_dir,'planning_units_model_error_by_country_',response_var_out,'.png'),
       plot = pu_preds.plot,
       width = 1200,
       height = 1080,
       scale = 3,
       units = 'px')


# ..........  XGBoost Vs GAM - difference map ====

pu_preds.plot <- ggplot(data = pu_preds.aggregated,
                        mapping = aes(x = PU_lon, 
                                      y = PU_lat,
                                      colour = xgb_vs_gam_diff)) +
  
  geom_point(size = 1, alpha = 0.6) +
  
  coord_equal() +
  
  # scale_color_gradient(low = "#CCCC00",
  #                      high = "#CC0000") +
  
  scale_colour_gradient2(low = '#FF2222', 
                         mid = '#cccccc',
                         high = '#2222FF') +
  
  labs(colour = 'XGBoost Vs GAM\ndifference') +
  
  theme_bw() +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        panel.grid = element_blank())

pu_preds.plot


ggsave(filename = paste0(out_dir,'planning_units_model_xgb_vs_gam_diff_',response_var_out,'.png'),
       plot = pu_preds.plot,
       width = 1200,
       height = 1080,
       scale = 3,
       units = 'px')


# ..........  XGBoost Vs GAM - difference scatter ====

pu_preds.plot <- ggplot(pu_preds.aggregated, 
                        aes(x = Coralcover.prop.predicted.GAM,
                            y = Coralcover.prop.predicted.mean,
                            colour = xgb_vs_gam_diff)) + 
  geom_point() + 
  theme_bw() +  
  scale_colour_gradient2(low = '#FF2222',
                         mid = '#cccccc',
                         high = '#2222FF') + 
  
  geom_abline(intercept = 0, slope = 1, colour = 'darkgrey') +
  
  geom_smooth(method = 'lm')

# coord_equal()

pu_preds.plot


ggsave(filename = paste0(out_dir,'planning_units_model_xgb_vs_gam_diff_scatter_',response_var_out,'.png'),
       plot = pu_preds.plot,
       width = 1080,
       height = 1080,
       scale = 3,
       units = 'px')



# ..........  PU predictions - GAM - Kenya ====
pu_preds.plot <- ggplot(data = pu_preds.aggregated %>% filter(Country == 'Kenya'),
                        mapping = aes(x = PU_lon, 
                                      y = PU_lat,
                                      colour = get(paste0(predict_var,'.GAM')))) +
  
  geom_point(size = 2, alpha = 0.6) +
  
  coord_equal() +
  
  labs(colour = 'Mean predicted\nmangrove cover\nGAM') +
  
  theme_bw() +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())

pu_preds.plot


ggsave(filename = paste0(out_dir,'planning_units_predictions_gam_kenya_',response_var_out,'.png'),
       plot = pu_preds.plot,
       width = 1200,
       height = 1080,
       scale = 3,
       units = 'px')



# ..........  PU predictions - GAM ====

pu_preds.plot <- ggplot(data = pu_preds.aggregated,
                        mapping = aes(x = PU_lon, 
                                      y = PU_lat,
                                      colour = get(paste0(predict_var,'.GAM')))) +
  
  geom_point(size = 2, alpha = 0.6) +
  
  coord_equal() +
  
  labs(colour = 'Mean predicted\nmangrove cover\nGAM') +
  
  theme_bw() +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())

pu_preds.plot


ggsave(filename = paste0(out_dir,'planning_units_predictions_gam_',response_var_out,'.png'),
       plot = pu_preds.plot,
       width = 1200,
       height = 1080,
       scale = 3,
       units = 'px')



# pu_preds.plot <- ggplot(data = pu_preds.aggregated,
#                         mapping = aes(x = PU_lon, 
#                                       y = PU_lat,
#                                       colour = get(paste0(predict_var,'.mean')))) +
#   
#   geom_point(size = 2, alpha = 0.6) +
#   
#   coord_equal() +
#   
#   labs(colour = 'Mean predicted\nmangrove cover\nXGBoost') +
#   
#   theme_bw() +
#   theme(axis.text.x = element_blank(),
#         axis.ticks.x = element_blank(),
#         axis.text.y = element_blank(),
#         axis.ticks.y = element_blank())
# 
# pu_preds.plot
# 
# 
# ggsave(filename = paste0(out_dir,'planning_units_predictions_xgb_',response_var_out,'.png'),
#        plot = pu_preds.plot,
#        width = 1200,
#        height = 1080,
#        scale = 3,
#        units = 'px')

# ..........  XGBoost Vs GAM - difference map - Kenya ====

pu_preds.plot <- ggplot(data = pu_preds.aggregated %>% filter(Country == 'Kenya'),
                        mapping = aes(x = PU_lon, 
                                      y = PU_lat,
                                      colour = xgb_vs_gam_diff)) +
  
  geom_point(size = 2, alpha = 0.6) +
  
  coord_equal() +
  
  # scale_color_gradient(low = "#CCCC00",
  #                      high = "#CC0000") +
  
  scale_colour_gradient2(low = '#FF2222', 
                         mid = '#cccccc',
                         high = '#2222FF') +
  
  labs(colour = 'XGBoost Vs GAM\ndifference') +
  
  theme_bw() +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        panel.grid = element_blank())

pu_preds.plot


ggsave(filename = paste0(out_dir,'planning_units_model_xgb_vs_gam_diff_kenya_',response_var_out,'.png'),
       plot = pu_preds.plot,
       width = 1200,
       height = 1080,
       scale = 3,
       units = 'px')




stop()
# SCRATCH ====
