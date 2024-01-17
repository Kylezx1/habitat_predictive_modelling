# habitat modelling pipeline

source("machine_learning_functions.r")

# ..... Single model run ====
start_time <- Sys.time()
xgb_model <- 
  xgboost_model_training_wrapper(input_data = model_data,
                                 response_var = response_var,
                                 replicate_unit_var = replicate_unit_var,
                                 id_vars = id_vars,
                                 features = features,
                                 features.one_hot = features.one_hot,
                                 response_type = response_type,
                                 train_test_split_val = train_test_split_val,
                                 train_watchlist_split_val = train_watchlist_split_val,
                                 
                                 xgb_booster = 'gbtree',
                                 global_objective = global_objective,  # Use for regression
                                 global_eval_metric = 'rmse',
                                 global_device = 'cpu',
                                 global_nthread = 12,
                                 
                                 tree_max_depth = 12,   # Adjust as needed
                                 tree_eta = 0.05,       # Learning rate
                                 tree_subsample = 0.5,
                                 tree_lambda = 0.5,
                                 tree_alpha = 0.5,
                                 tree_num_parallel_tree = 1,
                                 tree_method = 'hist',
                                 
                                 # xgb_booster = 'gbtree',
                                 # global_objective = custom_obj_beta_regression,
                                 # global_eval_metric = 'logloss',
                                 # maximize = TRUE,
                                 
                                 linear_lambda_bias = NULL,
                                 
                                 training_rounds = 20000,
                                 early_stopping_rounds = 200,
                                 print_progress = T,
                                 print_every_n = 2000)
time_end <- Sys.time() - start_time

# 0.01 ETA
#1: 0.058 rmse @ 3.4 mins
#2: 0.058 rmse @ 1,9 mins
#3: 0.058 rmse @ 2.4 mins w/efficient aggressive
#4: 0.058 rmse @ 2.2 mins w/efficient aggressive + nthread = 1

# 0.01 ETA, experimental version
#5: 0.063 rmse @ 0.26 mins w/efficient aggressive + nthread = 1 + device = cuda
#6: 0.063 rmse @ 0.18 mins w/efficient aggressive + nthread = 1 + device = cuda

# 0.01 ETA, tree method = 'approx'
#6: 0.063 rmse @ 1 mins w/efficient aggressive + nthread = 1 + device = cuda


xgb_model.predictions <- predict(object = xgb_model$xgboost_model,
                                 newdata = xgb_model$model_matrices.xgb$test,
                                 data = xgb_model$model_matrices.xgb$train)


dnorm(xgb_model.predictions)

xgb_model.predictions <- tibble(!!predict_var := xgb_model.predictions) %>%
  bind_cols(xgb_model$xgb_data_list$test[,c(response_var, replicate_unit_var, id_vars)]) %>%
  rename(!!actual_var := response_var)

xgb_model.predictions <- xgb_model.predictions %>%
  mutate(error = get(predict_var) - get(actual_var),
         error.abs = abs(error),
         error.pc = (error / get(actual_var)) * 100,
         error.abs.pc = abs(error.pc))


# ..... Multiple runs ====
# .......... Run models ====
xgb_models <- lapply(#mc.cores = 10,
                       1:n_runs, function(x) {
  
  print(paste0("running model ",x," of ", n_runs))
  
  xgb_model <- 
    xgboost_model_training_wrapper(input_data = model_data,
                                   response_var = response_var,
                                   replicate_unit_var = replicate_unit_var,
                                   id_vars = id_vars,
                                   features = features,
                                   features.one_hot = features.one_hot,
                                   response_type = response_type,
                                   train_test_split_val = train_test_split_val,
                                   train_watchlist_split_val = train_watchlist_split_val,
                                   
                                   xgb_booster = 'gbtree',
                                   global_objective = "reg:squarederror",  # Use for regression
                                   global_eval_metric = 'rmse',
                                   global_nthread = 18,
                                   global_device = 'cuda',
                                   
                                   
                                   tree_max_depth = 12,   # Adjust as needed
                                   tree_eta = 0.05,       # Learning rate
                                   tree_subsample = 0.5,
                                   tree_lambda = 0.5,
                                   tree_alpha = 0.5,
                                   tree_num_parallel_tree = 3,
                                   tree_method = 'hist',
                                   
                                   
                                   
                                   # xgb_booster = 'gbtree',
                                   # global_objective = custom_obj_beta_regression,
                                   # global_eval_metric = 'logloss',
                                   # maximize = TRUE,
                                   
                                   
                                   training_rounds = 20000,
                                   early_stopping_rounds = 200,
                                   print_progress = T
    )
  
  
  predictions <- predict(object = xgb_model$xgboost_model,
                         newdata = xgb_model$model_matrices.xgb$test,
                         data = xgb_model$model_matrices.xgb$train)
  
  predictions <- tibble(!!predict_var := predictions) %>%
    bind_cols(xgb_model$xgb_data_list$test[,c(response_var, replicate_unit_var, id_vars)]) %>%
    rename(!!actual_var := response_var)
  
  mean_actual <-  mean(predictions[[actual_var]])
  
  predictions <- predictions %>%
    mutate(run = x,
           mean_actual = mean_actual,
           error = get(predict_var) - get(actual_var),
           error.abs = abs(error),
           error.pc = (error / get(actual_var)) * 100,
           error.abs.pc = abs(error.pc),
           error_mean = get(predict_var) - mean_actual,
           error_mean.abs = abs(error_mean)) %>%
    relocate(run, all_of(id_vars))
  
  xgb_model[['predictions']] <- predictions
  
  shaps_x <- get_shap_values.xgboost(trained_xgb_model_object = xgb_model,
                                     response_var = response_var,
                                     predinteraction = F,
                                     problem_class = 'regression')
  shaps_x <- shaps_x %>% mutate(run = x)
  
  
  xgb_model[['shap_values']] <- shaps_x
  
  xgb_model
  
})


# .......... Model performance ====

xgb_models.predictions <- lapply(1:n_runs, function(x) {
  xgb_models[[x]]$predictions
}) %>% bind_rows() 


model_runs.performance.summary <- xgb_models.predictions %>%
  #pivot_longer(cols = c('accuracy', 'confidence'), names_to = 'metric') %>%
  group_by(across(c('run', 'country'))) %>%
  summarise(error.mean = mean(error),
            error.sd = sd(error),
            error_abs.mean = mean(error.abs),
            mean_only_error = mean(error_mean),
            mean_only_error.abs = mean(error_mean.abs)
            # confidence.mean = mean(confidence),
            # confidence.sd = sd(confidence)
  )


model_runs.performance.summary.mean_only_models <- model_runs.performance.summary %>%
  group_by(country) %>%
  summarise(mean_only_error = mean(mean_only_error),
            mean_only_error.abs = mean(mean_only_error.abs),
            median_only_error = median(mean_only_error),
            median_only_error.abs = median(mean_only_error.abs))


model_runs.performance.country_error <- model_runs.performance.summary %>%
  group_by(country) %>%
  summarise(model_error_rate.country = mean(mean_only_error.abs))


# .......... SHAP values ====

shap_values <- lapply(1:n_runs, function(x) {
  
  out <- xgb_models[[x]]$shap_values
  
  out
  
}) %>% bind_rows() 


features_wide <- shap_values %>%
  select(-BIAS, -observation_id, -data_type, -run,  -id_vars) %>%
  names()

shap_values.long <- shap_values %>%
  pivot_longer(cols = features_wide,
               names_to = 'feature',
               values_to = 'shap') %>%
  mutate(shap_abs = abs(shap)) %>%
  arrange(-shap_abs)



# .......... Predict to PUs ====
if(exists('pu_data')) {

  
  pu_data2 <- set_up_model_data(input_data = pu_data,
                               response_var = response_var,
                               response_type = response_type,
                               features = features,
                               id_vars = id_vars,
                               features.one_hot = features.one_hot,
                               replicate_unit_var = NA,
                               train_test_split_val = 0,
                               train_watchlist_split_val = 0)


  train_features.data <- xgb_models[[x]]$xgb_data_list$train[0,]
  train_features.names <- xgb_models[[x]]$xgboost_model$feature_names
  
  
  pu_data2[[1]] <- pu_data2[[1]] %>%
    bind_rows(train_features.data) %>%
    select(train_features.names, response_var)
  
  pu_data.xgb_matrix <- convert_data_to_xgb_matrix(xgb_data_list = pu_data2[1],
                                                   features = train_features.names,
                                                   response_var = response_var,
                                                   response_type = response_type,
                                                   missing = NA) %>%
    .[[1]]
  
  
  pu_predictions <- lapply(1:n_runs, function(x) {
  
    model_x <- xgb_models[[x]]$xgboost_model
    data_x <- xgb_models[[x]]$model_matrices.xgb$train
    preds_x <- predict(object = model_x,
                       data = data_x,
                       newdata = pu_data.xgb_matrix)
  
    out <- tibble(!!predict_var := preds_x,
                  model_run = x) %>%
      bind_cols(pu_data)
  
    out
  
  }) %>% bind_rows()
  
  
  pu_preds.aggregated <- pu_predictions %>%
    group_by(across(any_of(c(id_vars)))) %>%
    summarise(!!paste0(predict_var,'.mean') := mean(get(predict_var)),
              !!paste0(predict_var,'.sd') := sd(get(predict_var))
    ) %>%
    ungroup() %>%
    left_join(model_runs.performance.country_error) %>%
    mutate(pu_lat = as.numeric(pu_lat),
           pu_lon = as.numeric(pu_lon))
  
  
  pu_preds_pivot_vars <- c(paste0(predict_var,'.mean'),
                           paste0(predict_var,'.sd'),
                           'model_error_rate.country')
  
  pu_preds.aggregated

}

# ..... Savin' data ====

if(save_output_data) {
  
  if(exists('model_runs.performance.country_error')) {
    
    write_csv(x = model_runs.performance.country_error,
              file = paste0(out_dir,'model_performance_by_country.csv'))

  }
  
  if(exists('model_runs.performance.summary')) {
    
    write_csv(x = model_runs.performance.summary,
              file = paste0(out_dir,'model_performance_summary.csv'))

  }
  
  
  if(exists('xgb_models.predictions')) {
    
    write_csv(x = xgb_models.predictions,
              file = paste0(out_dir,'model_predictions_raw.csv'))
  
  }
  
  
  if(exists('shap_values')) {
    
    write_csv(x = shap_values,
              file = paste0(out_dir,'shap_values.csv'))
      
  }
  
  
  if(exists('pu_preds.aggregated')) {
    
    write_csv(x = pu_preds.aggregated,
              file = paste0(out_dir,'planning_unit_predictions.csv'))
    
  }
  
}

