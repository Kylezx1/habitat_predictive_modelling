# XGBpoost functions

# XRF prediction API

# Author: Kyle Zawada 2023
# Contact: kyle.zawada@mq.edu,au OR kylezx1@gmail.com
# GitHub: kylezx1 


# DESCRIPTION:
# This API/set of functions allows the user to build and test predictive models

# TODO: Add documentation to functions


# Set up  ====

# Libraries
library(tidyverse)
library(caret)

library(xgboost)

library(neuralnet)
library(geodist)

library(grid)
library(gridExtra)

library(openssl)

# library(parallel)


# Functions ====
# ..... Generic ====

# Function to interleave colors
interleave_colors <- function(num_colors, num_splits,
                              begin = 0, end = 1) {
  library(viridis)
  
  # Generate the colors using the viridis palette
  colors <- viridis(num_colors, begin = begin, end = end)
  
  # Calculate the number of colors per split
  colors_per_split <- ceiling(num_colors / num_splits)
  
  # Create a matrix of colors with the specified number of splits
  color_matrix <- matrix(colors[1:(colors_per_split * num_splits)], nrow = colors_per_split)
  
  
  # Remove 
  
  # Interleave the colors from each split
  interleaved_colors <- as.vector(t(color_matrix)) %>% na.omit()
  
  return(interleaved_colors)
}



split_data_by_index <- function(model_dataframe, y_index, replicate_unit_var) {
  
  y_data <- model_dataframe
  
  # Pull out replicate vars from selection indexes
  if(!is.na(replicate_unit_var)) {
    
    y_data <- y_data[y_index,] %>% dplyr::select(matches(replicate_unit_var)) %>% distinct()
    
  } else {
    
    y_data <- y_data[y_index,]
    
  }
  
  # Add column for selection
  y_data <- y_data %>% mutate(split = 'y')
  # join back to main dataframe
  model_dataframe_2 <- model_dataframe %>% full_join(y_data)
  # Fill NAs
  model_dataframe_2 <- model_dataframe_2 %>% mutate(split = ifelse(is.na(split), 'x', 'y'))
  # Split and remove split column
  out <- list(x = model_dataframe_2 %>% filter(split == 'x') %>% dplyr::select(-split),
              y = model_dataframe_2 %>% filter(split == 'y') %>% dplyr::select(-split))
  
  return(out)
  
}


# With replicate grouping
split_data <- function(model_dataframe, 
                       response_var = NULL, 
                       replicate_unit_var = 'sample_id',
                       split_type = c('proportion', 'partition', 'group_n'),
                       split_val = 0.2, 
                       group_n.replace = F,
                       y_name = 'test', 
                       x_name = 'train') {
  
  if(split_val == 0){
    
    out <- list(model_dataframe, tibble())
    names(out) <- c(x_name, y_name)
    return(out)
    
  }
  
  # Generate selection indexes
  if(split_type[1] == 'proportion') {
    
    y_index <- model_dataframe %>% 
      mutate(row_number = row_number()) %>% 
      group_by(get(response_var)) %>%
      slice_sample(prop = split_val)
    y_index <- y_index$row_number
    
  } else if(split_type[1] == 'partition') {
    
    if(length(response_var) > 1) { 
      print('warning, multiple response variables provided, data will be partitioned based on the first response variable provided. Switch to "group_n" to sample across all responses')  
    }
    
    y_index <- createDataPartition(model_dataframe[[response_var[1]]],
                                   p = split_val,
                                   times = 1)
    y_index <- y_index$Resample1
    
  } else if(split_type[1] == 'group_n') {
    
    y_index <- model_dataframe %>% 
      mutate(row_number = row_number()) %>%
      group_by(get(response_var)) %>%
      slice_sample(n = split_val, replace = group_n.replace)
    
    y_index <- y_index$row_number
    
  }
  
  # Split data
  out <- split_data_by_index(model_dataframe = model_dataframe,
                             y_index = y_index,
                             replicate_unit_var = replicate_unit_var)
  
  names(out) <- c(x_name, y_name)
  
  return(out)
  
}



probabilities_to_classes <- function(probability_table, response_var, id_vars) {
  
  out <- probability_table %>% mutate(pred_number = row_number())
  
  predicted_classes <- out %>%
    pivot_longer(cols = -any_of(c(response_var, id_vars, 'pred_number')),
                 names_to = paste0(response_var,'.predicted')) %>%
    group_by(pred_number) %>%
    arrange(-value) %>%
    slice_head(n = 1) #%>%
  
  errors <- ifelse(predicted_classes[[response_var]] == predicted_classes[[paste0(response_var,'.predicted')]], 0, 1)
  
  predicted_classes[['class_error']] <- errors
  
  out <- full_join(out, predicted_classes, by = unique(c('pred_number', response_var))) %>%
    dplyr::select(-pred_number) %>%
    rename(!!paste0(response_var,'.actual') := response_var,
           predicted_class_probability = value)
  
  return(out)
  
}


balance_factor_by_smallest_class <- function(data, class_var, bootstrap = F, smallest_class_n_override = NA) {
  
  if(bootstrap == T) { 
    print('Warning, bootstrap option should be false. Bootstrap = T causes unstable dataset sizes')
    print('this option will be removed in the future')
  }
  
  
  class_count <- data %>% 
    group_by(across(class_var)) %>%
    summarise(n = n()) %>%
    arrange(n)
  smallest_class_n <- class_count$n[1]
  
  if(!is.na(smallest_class_n_override)) {
    
    if(smallest_class_n_override > smallest_class_n) {
      
      print(paste0('smallest class ovverride is:',smallest_class_n_override))
      print(paste0('smallest class count in the data is:',smallest_class_n))
      print(paste0('set smallest_class_n_override <= ',smallest_class_n))
      stop()
      
    } else {
      
      smallest_class_n <- smallest_class_n_override
      
    }
    
  }
  
  out <- data %>%
    group_by(across(class_var)) %>% 
    slice_sample(n = smallest_class_n, replace = bootstrap) %>%
    ungroup()
  
  return(out)
  
}


set_up_model_data <- function(input_data, 
                              response_var,
                              features,
                              id_vars,
                              replicate_unit_var = 'sample_id',
                              train_test_split_type = 'partition',
                              train_watchlist_split_type = 'partition',
                              response_type = c('classification', 'regression'),
                              train_test_split_val = 0.2,
                              train_watchlist_split_val = 0.3,
                              split_function = split_data,
                              balance_factor_by_smallest_class_bool = T,
                              smallest_class_n_override = NA,
                              bootstrap = NA,
                              print_dataset_sizes = F
) {
  
  
  if(!is.na(bootstrap)) print('Warning bootstrap option is no longer functional due to causing issues with imbalanced datasets. Remove this from the function call')
  
  
  # Filter data
  model_data_2 <- input_data %>% dplyr::select(c(response_var, replicate_unit_var, id_vars, features) %>% na.omit())
  
  
  # Convert response to factor 
  if(response_type[1] == 'classification') {
    model_data_2 <- model_data_2 %>% mutate(across(response_var, factor))
  }
  
  
  # Select one replicate_unit at random
  if(!is.na(replicate_unit_var)) {
    model_data_2 <- model_data_2 %>% group_by(across(replicate_unit_var)) %>% slice_sample(n = 1)
  }
  
  
  # Balance classes
  if(response_type[1] == 'classification' & balance_factor_by_smallest_class_bool == T) {
    
    model_data_2 <- 
      balance_factor_by_smallest_class(data = model_data_2,
                                       class_var = response_var,
                                       smallest_class_n_override = smallest_class_n_override)
  }
  
  
  # Split data
  train_test <- split_data(model_dataframe = model_data_2,
                           response_var = response_var,
                           replicate_unit_var = replicate_unit_var,
                           split_val = train_test_split_val,
                           split_type = train_test_split_type,
                           y_name = 'test',
                           x_name = 'train')
  
  
  train_watch <- split_data(model_dataframe = train_test$train,
                            response_var = response_var,
                            replicate_unit_var = replicate_unit_var,
                            split_val = train_watchlist_split_val,
                            split_type = train_watchlist_split_type,
                            y_name = 'watchlist',
                            x_name = 'train')
  
  
  # Combine into single list
  out <- list(train = train_watch$train,
              watchlist = train_watch$watchlist,
              test = train_test$test)
  
  out <- lapply(out, ungroup)
  
  
  # Print final dataset sizes
  if(print_dataset_sizes) { lapply(out, function(x) { x[[response_var]] %>% table()}) }
  
  
  return(out)
  
}



# ..... WrapperiseR ====

filter_dots <- function(dots, function_x) {
  passed <- names(dots)
  out <- dots[which(passed %in% names(formals(function_x)))]
  return(out)
}


function_with_dots <- function(function_x, dots) {
  args <- filter_dots(dots, function_x)
  out <-  do.call(what = function_x,
                  args = args)
  return(out)
  
}

wrapperise_functions <- function(function_list,
                                 target_outputs,
                                 
                                 print_arguments_bool = F,
                                 
                                 ...) {
  
  
  # Print function list arguments if needed
  if(print_arguments_bool) {
    
    all_args <- lapply(function_list, function(x) { f <- formals(x[[1]]) })
    
    print('printing and returning all arguments and defaults for functions in the wrapper.
           Set "print_arguments_bool = F" to stop this behaviour')
    print(all_args)
    
    return(all_args)
    
  }
  
  # Execute functions in sequence
  t_start <- Sys.time()
  dots <- as.list(match.call())
  # dots <- alist(match.call())
  dots <- dots[grepl("function_list|target_outputs|print_arguments_bool|\\.\\.\\.", names(dots)) == F]
  
  for(x in 1:length(function_list)) {
    
    run_x <- function_list[[x]]
    
    function_x <- run_x[[1]][[1]]
    output_x <- run_x[[2]]
    message_x <- run_x[[3]]
    
    print(message_x)
    out <- function_with_dots(function_x = function_x %>% eval(),
                              dots = dots)
    
    for(y in 1:length(output_x)) {
      
      output_x_y <- output_x[[y]]
      
      output_y <- output_x_y[1]
      out_find_y <-  output_x_y[2]
      
      out_y <- paste0('out',out_find_y)
      
      eval(parse(text = "dots[[output_y]] <- eval(parse(text = out_y))"))
      
    }
    
    t_end <- Sys.time() - t_start
    t_end %>% round(2) %>% print()
    
  }
  
  print('getting target outputs')
  
  out_dots <- lapply(target_outputs, function(x) dots[[x]])
  names(out_dots) <- target_outputs
  
  t_end <- Sys.time() - t_start
  t_end %>% round(2) %>% print()
  
  return(out_dots)
  
}


generate_wrapper_function <- function(function_args_alist,
                                      function_list,
                                      target_outputs,
                                      ...) {
  
  out <- wrapperise_functions
  
  formals(out) <- c(function_args_alist, formals(out))
  
  formals(out)$target_outputs <- target_outputs
  formals(out)$function_list <- function_list
  
  
  return(out)
  
}





# ..... XGBoost library ====
### create_xgb_matrix ###

# XGBoost requires data in a specific matrix format, this function generates that matrix

# model_dataframe: a dataframe containing the response variable and predictors
# response_var: a string denoting the response variable. e.g. 'abundance'

# output: an xgb.DMatrix object to be passed into xgb.train

create_xgb_matrix <- function(model_dataframe, 
                              features,
                              response_var, 
                              response_type = c('classification', 'regression')) {
  
  if(nrow(model_dataframe) == 0) { return() }
  
  xgb_features <- model_dataframe %>% 
    dplyr::select(all_of(features)) %>% 
    as.matrix()
  
  xgb_labels <- model_dataframe %>% 
    dplyr::select(response_var) %>% 
    .[,1] %>% 
    as.matrix()
  
  if(response_type[1] == 'classification') { xgb_labels <- as.numeric(factor(xgb_labels))-1 }
  
  out <- xgb.DMatrix(data = xgb_features, label = xgb_labels)
  
  return(out)
}


convert_data_to_xgb_matrix <- function(xgb_data_list, features, response_var, response_type = 'classification') {
  
  out <- lapply(xgb_data_list,  # for each thing in this list
                create_xgb_matrix,  # do this
                response_var = response_var,
                response_type = response_type,
                features = features)  # extra arguments to the above function
  
  return(out)
  
}


set_up_parameters.xgboost <- function(xgb_booster = c('gbtree','gblinear'),  # tree or linear booster?
                                      
                                      xgb_num_class,
                                      
                                      maximize = NULL,
                                      
                                      global_eval_metric = 'mlogloss',  # the metric to judge model performance, here multi-class log-loss
                                      global_objective = 'multi:softprob',  # the objective function, defines the problem class additional option, here multi-class and softmax with outputs as probabilities)
                                      global_base_score = 0.5,
                                      global_nthread = 12,
                                      global_device = 'cuda',
                                      
                                      
                                      tree_eta = 0.3,  # the learning rate, default = 0.3
                                      tree_gamma = 0.01, # minimum loss to make further partition
                                      tree_max_depth = 6,  # maximum tree depth, default = 6
                                      tree_min_child_weight = 1,
                                      tree_subsample = 1,
                                      tree_colsample_bytree = 1,
                                      tree_lambda = 1,
                                      tree_alpha = 0,
                                      tree_num_parallel_tree = 1,
                                      tree_method = 'hist',
                                      
                                      linear_lambda = 0,
                                      linear_lambda_bias = 0,
                                      linear_alpha = 0) {
  
  # Global parameters
  xgb_params_global <- list(booster = xgb_booster, 
                            eval_metric = global_eval_metric, 
                            objective =  global_objective,
                            nthread = global_nthread,
                            device = global_device)
  
  # Tree parameters
  xgb_params_tree <- list(num_class = xgb_num_class,
                          eta = tree_eta, 
                          gamma =  tree_gamma,
                          max_depth = tree_max_depth, 
                          min_child_weight = tree_min_child_weight,
                          subsample = tree_subsample,
                          colsample_bytree = tree_colsample_bytree,
                          lambda = tree_lambda,
                          alplha = tree_alpha,
                          tnum_parallel_tree = tree_num_parallel_tree,
                          tree_method = tree_method)
  
  # Linear parameters
  xgb_params_linear <- list(lambda = linear_lambda, lambda_bias = linear_lambda_bias, alpha = linear_alpha)
  
  
  # combine the global and tree parameter lists
  if(xgb_booster[1] == 'gbtree') {
    out <- c(xgb_params_global, xgb_params_tree) 
  } else if(xgb_booster[1] == 'gblinear') {
    out <- c(xgb_params_global, xgb_params_linear) 
  }
  
  return(out)
  
}


train_model.xgboost <- function(train_data,
                                watchlist_data = NULL,
                                params = NULL,
                                training_rounds = 20000,
                                early_stopping_rounds = 50,
                                print_progress = T,
                                maximize = NULL,
                                ...) {
  
  verbose <- ifelse(print_progress, 1, 0) 
  
  out <- xgb.train(data = train_data,
                   watchlist = list('watchlist' = watchlist_data),
                   params = params,
                   nround = training_rounds,
                   early_stopping_rounds = early_stopping_rounds,
                   verbose = verbose,
                   maximize = maximize,
                   ...)
  
  return(out)
  
}


evaluate_test_set.xboost.multiclass <- function(xgboost_model,
                                                test_data, 
                                                test_xgb_matrix, 
                                                response_var, 
                                                id_vars) {
  
  # Predict for the test set as a vector of probabilities
  test_predictions <- predict(xgboost_model, newdata = test_xgb_matrix)
  
  # Back calculate the number of classes
  n_classes <- length(test_predictions)/nrow(test_data)
  
  # Get the class names
  class_names <- levels(test_data[[response_var]])
  
  if(is.null(class_names)) class_names <- 'predictions'
  
  # Convert the vector of probabilities into a dataframe
  test_preds_frame <- test_predictions %>%
    matrix(nrow = nrow(test_data),
           ncol = n_classes,
           byrow = T) %>%
    as_tibble()
  
  # This renames the columns to match the categories
  names(test_preds_frame) <- class_names
  
  # Combine with the actuals
  test_preds_and_actuals <- test_preds_frame %>% mutate(!!response_var := test_data[[response_var]])
  
  # Convert predicted probabilities to a predicted class
  out <- probabilities_to_classes(probability_table = test_preds_and_actuals, 
                                  response_var = response_var,
                                  id_vars = id_vars)
  
  out <- test_data %>% dplyr::select(all_of(c(id_vars))) %>% bind_cols(out)
  
  return(out)
  
}


get_shap_values.xgboost <- function(trained_xgb_model_object,
                                    response_var,
                                    predinteraction = F, 
                                    problem_class = c('multiclass', 'twoclass', 'regression')) {
  
  class_levels <- levels(trained_xgb_model_object$xgb_data_list$train[[response_var]])
  # class_levels <- trained_xgb_model_object$xgb_data_list$train[[response_var]] %>% unique()
  
  shaps_all <- lapply(1:3, function(x) {
    
    xgb_matrix_x <- trained_xgb_model_object$model_matrices.xgb[x]
    xgb_data_x <- trained_xgb_model_object$xgb_data_list[x]
    
    data_type <- names(xgb_matrix_x)
    
    if(nrow(xgb_data_x[[1]]) == 0) { return() }
    
    print(paste0('calculating SHAP values for: ',data_type))
    
    shaps <- predict(object = trained_xgb_model_object$xgboost_model, 
                     newdata = xgb_matrix_x[[1]], 
                     predcontrib = TRUE, 
                     approxcontrib = F, 
                     predinteraction = predinteraction)
    
    shaps <- bind_cols(xgb_data_x[[1]] %>% dplyr::select(id_vars), shaps) %>%
      mutate(data_type = data_type,
             observation_id = paste0(data_type,'_',row_number()))

    shaps
    
    
    # preds <- evaluate_test_set.xboost.multiclass(xgboost_model = trained_xgb_model_object$xgboost_model,
    #                                              test_data = xgb_data_x[[1]],
    #                                              test_xgb_matrix = xgb_matrix_x[[1]],
    #                                              response_var = response_var,
    #                                              id_vars = id_vars)
    # 
    # preds <- preds %>% mutate(observation_id = paste0(data_type,'_',row_number()))
    # 
    # 
    # 
    # 
    # if(problem_class[1] == 'multiclass') {
    #   
    #   shaps <- lapply(1:length(shaps), function(y) {
    #     
    #     shaps_y <- shaps[[y]] %>% as_tibble()
    #     names(shaps_y) <- paste0(names(shaps_y),'.shap')
    #     
    #     
    #     preds %>% 
    #       mutate(shap_class = class_levels[y],
    #              data_type = data_type) %>% 
    #       bind_cols(shaps_y)
    #     
    #   }) %>% bind_rows() %>% mutate(data_type = data_type) %>% relocate(data_type)
    #   
    # } else {
    #   
    #   shaps <- lapply(1:length(shaps), function(y) {
    #     
    #     shaps_y <- shaps[[y]] %>% as_tibble()
    #     names(shaps_y) <- paste0(names(shaps_y),'.shap')
    #     
    #     # TODO check this works
    #     shaps <- preds %>% 
    #       mutate(shap_class = class_levels[y],
    #              data_type = data_type) %>% 
    #       bind_cols(shaps_y)
    #     
    #   }) %>% bind_rows() %>% mutate(data_type = data_type) %>% relocate(data_type)
    #   
    # }
    
  }) %>% bind_rows()
  
  # 
  # 
  # WIP need to convert multiclass log odds to probability.
  # logodds_to_prob <- function(logodds) { return(sum(exp(logodds)/exp(logodds))) }
  # 
  # x <- 1
  # logodds_x <- lapply(shaps, function(y) y[x,])
  # pred_x <- preds[x,]
  # 
  # y <- 1
  # base_val_y <- shaps[[y]][x,"BIAS"]
  # pred_val_y <- base_val_y + sum(shaps[[y]][x,1:(ncol(shaps[[y]])-1)])
  # converted_prob_val <- logodds_to_prob(logodds_x[[y]])
  # 
  # 
  # if(problem_class[1] == 'multiclass') {
  #   
  #   shaps_2 <- lapply(shaps, function(x) {
  #     
  #     logodds_to_prob(x)
  #     
  #   })
  #
  #}
  
  
  return(shaps_all)
  
}


one_hot_row_sums <- function(dataframe, categorical_columns) {
  
  if(is.na(categorical_columns)) { return(dataframe) }
  
  out <- lapply(categorical_columns, function(col) {
    sums <- dataframe %>% dplyr::select(matches(col)) %>% rowSums()
    out <- dataframe %>% dplyr::select(-matches(categorical_columns))
    out[col] <- sums
    out
  }) 
  
  out <- suppressMessages(reduce(out, full_join))
  
  return(out)
}


one_hot_decode <- function(dataframe, categorical_columns) {
  
  if(is.na(categorical_columns)) { return(dataframe) }
  
  out <- lapply(categorical_columns, function(col) {
    dataframe  %>%
      pivot_longer(cols = matches(paste0('one_hot_', col, '...')), names_to = 'one_hot_category_one_hot') %>%
      filter(value == 1) %>%
      mutate(one_hot_category_one_hot = gsub(pattern = paste0('one_hot_', col, '...'), replacement = '', x = one_hot_category_one_hot)) %>%
      rename(!!col := one_hot_category_one_hot) %>%
      dplyr::select(-value, -matches('one_hot'))
  }) 
  
  out <- suppressMessages(reduce(out, full_join))
  
  return(out)
  
}


one_hot_decode_feature_importance <- function(importance_table, categorical_vars) {
  
  if(is.na(categorical_vars)) { return(importance_table) }
  
  id_vars <- names(importance_table)
  id_vars <- id_vars[-which(id_vars %in% c('feature', 'importance'))]
  
  out <- importance_table %>%
    pivot_wider(id_cols = id_vars, names_from = feature, values_from = importance) %>%
    one_hot_row_sums(dataframe = ., categorical_columns = categorical_vars) %>%
    pivot_longer(cols = -matches(c(id_vars, 'importance')), names_to = 'feature', values_to = 'importance')
  
  return(out)
  
}


mean_scale <- function(x) {  (x - mean(x, na.rm=TRUE)) / sd(x, na.rm=TRUE) }


process_shap_values.pivot_long <- function(shap_values) {
  
  out <- shap_values %>% pivot_longer(cols = matches(c('.shap')), names_to = 'feature', values_to = 'SHAP')
  
  return(out)
  
}


summarise_shap_values.feature_importance <- function(shap_values.long) {
  
  out <- shap_values.long %>% 
    group_by(data_type, shap_class, feature) %>% 
    summarise(mean_SHAP = mean(SHAP), 
              mean_absolute_SHAP = mean(abs(SHAP)),
              sum_absolute_SHAP = sum(abs(SHAP)))
  
  return(out)
  
}


process_multirun_results <- function(multirun_results) {
  
  response_var <- multirun_results$response[1]
  
  out <- multirun_results %>%
    mutate(model_aggregation = ifelse(get(response_var) == 'all', 'all models', 'breakdown')) %>%
    mutate(overall_accuracy = ((1-overall_mean_class_error) * 100) %>% round(),
           sd_accuracy = (sd_class_error * 100) %>% round(),
           expected_accuracy = ((1-random_error) * 100) %>% round(),
           model_uplift.percent = (overall_accuracy-expected_accuracy)/expected_accuracy,
           model_uplift.fold = overall_accuracy/expected_accuracy)
  
  return(out)
  
}


plot_multirun_results <- function(multirun_results,
                                  plot.colours.low = '#CEDAFC',
                                  plot.colours.high = '#0F4BEB') {
  
  response_var <- multirun_results$response[1]
  random_accuracy <- multirun_results$expected_accuracy[1]
  
  results.plot <- ggplot(data = multirun_results,
                         aes(y = reorder(get(response_var), overall_accuracy),
                             x = overall_accuracy,
                             xmax = overall_accuracy + sd_accuracy,
                             xmin = overall_accuracy - sd_accuracy,
                             fill = overall_mean_confidence)) +
    geom_hline(yintercept = 'all', size = 4, colour = '#FFD2CD', alpha = 0.3) +
    geom_vline(xintercept = random_accuracy,
               linetype = 'dashed', size = 2, colour = 'darkgrey') +
    geom_errorbarh(height = 0.2) +
    geom_point(size = 5, shape = 21) +
    geom_point(data = multirun_results %>% filter(get(response_var) == 'all'), shape = 22, size =7) +
    scale_x_continuous(limits = c(0,100)) +
    scale_fill_gradient(low =plot.colours.low, high = plot.colours.high, limits = c(random_accuracy/100, 1)) +
    
    facet_grid(model_aggregation~., switch = 'y', scales = 'free_y', space='free') +
    
    labs(y = response_var) +
    
    theme_bw()
  
  
  return(results.plot)
  
}


plot_shap_values.shap_ditributions.classification <- function(shap_values.long, 
                                                              bw = 0.25,
                                                              filter_non_important_features = T,
                                                              minimum_abs_shap = 0) {
  
  if(filter_non_important_features) {
    shap_values.long <- shap_values.long %>% filter(mean_absolute_SHAP != 0)
    
  }
  
  plot <- ggplot(data = shap_values.long %>% filter(mean_absolute_SHAP > minimum_abs_shap),
                 aes(y = mean_SHAP, x = data_type, colour = shap_class)) +
    geom_hline(yintercept = 0, size = 1, colour = 'grey') +
    geom_boxplot(alpha = 0, size = 1) +
    theme_minimal() +
    scale_colour_viridis_d(option = 'D', begin = 0.2, end = 0.8) +
    facet_wrap(~reorder(feature,-mean_absolute_SHAP))
  
  # plot <- ggplot(data = shap_values.long %>% filter(mean_absolute_SHAP > 0.1), 
  #                aes(y = mean_absolute_SHAP, 
  #                    ymax = mean_absolute_SHAP + sd_absolute_SHAP,
  #                    ymin = mean_absolute_SHAP - sd_absolute_SHAP,
  #                    x = reorder(feature, -mean_absolute_SHAP),
  #                    size = mean_absolute_SHAP)) + 
  #   geom_hline(yintercept = 0, colour = 'grey') +
  #   geom_errorbar(size = 1) +
  #   geom_point(alpha = 1) + 
  #   theme_minimal() +
  #   theme(axis.text.x = element_text(angle = 45, vjust = 0.5))
  
  
  return(plot)
  
}



# OLD SHAP Function
# process_shap_values <- function(shap_values, categorical_features, train_data, keep_one_hot_bool = F, id_vars) {
#   
#   shap_values_2 <- one_hot_row_sums(shap_values[[1]], categorical_features)
#   shap_dataframe_melted <- shap_values_2 %>% 
#     as_tibble() %>% 
#     mutate(row_id = row_number()) %>% 
#     pivot_longer(cols = !matches(c('row_id', 'BIAS')), names_to = 'feature', values_to = 'SHAP')
#   
#   shap_summary <- shap_dataframe_melted %>% group_by(feature) %>% summarise(mean_SHAP = mean(SHAP), mean_absolute_SHAP = mean(abs(SHAP)))
#   
#   train_data_melted <- train_data %>% 
#     mutate(row_id = row_number()) %>% 
#     one_hot_decode(categorical_columns = categorical_features)
#   
#   if(!is.na(categorical_features)) {
#     
#     train_data_melted_categorical <- train_data_melted %>% 
#       dplyr::select(all_of(categorical_features), 'row_id') %>% 
#       pivot_longer(cols = !matches(c('row_id')), names_to = 'feature', values_to = 'value')
#     
#   }
#   
#   train_data_melted_continuous <- train_data_melted %>% 
#     pivot_longer(cols = !matches(c('row_id', id_vars)), names_to = 'feature', values_to = 'value') %>%
#     group_by(feature) %>%
#     mutate(value_scaled = mean_scale(value))
#   
#   if(!is.na(categorical_features)) {
#     
#     shap_table_categorical <- shap_dataframe_melted %>% 
#       right_join(train_data_melted_categorical, by = c('row_id', 'feature')) %>%
#       left_join(shap_summary, by = 'feature')
#     
#   } else {
#     
#     shap_table_categorical <- tibble()
#     
#   }
#   
#   shap_table_continuous <- shap_dataframe_melted %>% 
#     right_join(train_data_melted_continuous, by = c('row_id', 'feature')) %>%
#     left_join(shap_summary, by = 'feature')
#   
#   out <- list('continuous_shap_table' = shap_table_continuous, 'categorical_shap_table' = shap_table_categorical)
#   
#   return(out)
# 
# }


# .......... XGBoost model training wrapper ====
# ............... Experimental version ====
xgboost_model_training_wrapper <- 
  generate_wrapper_function(function_args_alist = alist(input_data=, 
                                                        response_var=,
                                                        features=,
                                                        id_vars=,
                                                        threshold=,
                                                        replicate_unit_var = replicate_unit_var,
                                                        train_test_split_type = 'partition',
                                                        train_watchlist_split_type = 'partition',
                                                        response_type = 'classification',
                                                        train_test_split_val = 0.2,
                                                        train_watchlist_split_val = 0.2,
                                                        balance_factor_by_smallest_class = T
  ),
  function_list = list(list(alist(set_up_model_data), list(c("xgb_data_list", '')), 'setting up model for xgboost'),
                       
                       list(alist(convert_data_to_xgb_matrix), list(c("model_matrices.xgb", ''),
                                                                    c('train_data', '$train'),
                                                                    c('watchlist_data', '$watchlist'),
                                                                    c('test_xgb_matrix', '$test')), 'generating xgboost matrices'),
                       
                       list(alist(set_up_parameters.xgboost), list(c('params', '')), 'generating xgboost parameters'),
                       
                       list(alist(train_model.xgboost), list(c('xgboost_model', '')), 'training xgboost model')),
  target_outputs = c('xgboost_model', 'model_matrices.xgb', 'xgb_data_list'))


# ............... Working version ====
xgboost_model_training_wrapper <- 
  function(input_data, 
           response_var,
           replicate_unit_var,
           id_vars = NA,
           features,
           response_type = c('classification', 'regression'),
           train_test_split_val = 0.2,
           train_watchlist_split_val = 0.3,
           
           print_arguments_bool = F,
           ...) {
    
    if(print_arguments_bool) {
      
      all_args <- list(set_up_model_data = formals(set_up_model_data),
                       convert_data_to_xgb_matrix = formals(convert_data_to_xgb_matrix),
                       set_up_parameters.xgboost = formals(set_up_parameters.xgboost),
                       train_model.xgboost = formals(train_model.xgboost),
                       evaluate_test_set.xboost.multiclass = formals(evaluate_test_set.xboost.multiclass))
      print('printing and returning all arguments and defaults for functions in the wrapper.
          Set "print_arguments_bool = F" to stop this behaviour')
      print(all_args)
      return(all_args)
      
    }
    
    
    dots <- list(...)  # These can be any additional arguments to any function within the wrapper
    dots <- c(list(input_data = input_data, 
                   response_var = response_var,
                   replicate_unit_var = replicate_unit_var,
                   id_vars = id_vars,
                   features = features,
                   response_type = response_type,
                   train_test_split_val = train_test_split_val,
                   train_watchlist_split_val = train_watchlist_split_val
    ),
    dots)
    
    # append_output_to_dots <- function(function_x, dots, output_name) {
    # 
    #   output <- function_with_dots(function_x = function_x, dots = dots)
    #   output <- list(output)
    #   names(output) <- output_name
    # 
    #   out <- c(dots, output)
    # 
    # 
    # }
    
    t_start <- Sys.time()
    print('setting up model for xgboost')
    xgb_data_list <- function_with_dots(function_x = set_up_model_data, 
                                        dots = dots)
    dots[['xgb_data_list']] <- xgb_data_list
    
    
    print('generating xgboost matrices')
    model_matrices.xgb <- function_with_dots(function_x = convert_data_to_xgb_matrix, 
                                             dots = dots)
    dots[['model_matrices.xgb']] <- model_matrices.xgb
    dots[['train_data']] <- model_matrices.xgb$train
    dots[['watchlist_data']] <- model_matrices.xgb$watchlist
    dots[['test_data']] <- xgb_data_list$test
    dots[['test_xgb_matrix']] <- model_matrices.xgb$test
    dots[['xgb_num_class']] <- length(unique(input_data[[response_var]]))
    
    print('generating xgboost parameters')
    params <- function_with_dots(function_x = set_up_parameters.xgboost, 
                                 dots = dots)
    params[['num_class']] <- NULL
    dots[['params']] <- params
    
    
    print('training xgboost model')
    xgboost_model <- function_with_dots(function_x = train_model.xgboost, 
                                        dots = dots)
    dots[['xgboost_model']] <- xgboost_model
    
    t_end <- Sys.time() - t_start
    print(paste0('finished in: ', round(t_end, 3), ' seconds'))
    
    return(list(xgboost_model = xgboost_model,
                # test_set_results = test_set_results,
                model_matrices.xgb = model_matrices.xgb,
                xgb_data_list = xgb_data_list))
    
  }


# This function isn't working ====
iterate_train_and_test.xgboost <- function(model_data,
                                           response_var,
                                           id_vars,
                                           replicate_unit_var,
                                           features,
                                           response_type,
                                           n_runs = 100,
                                           train_test_split_val = 0.1,
                                           train_watchlist_split_val = 0.2) {
  
  runs <- lapply(1:n_runs, function(x) {
    
    train_test_split_val_x <<- train_test_split_val
    train_watchlist_split_val_x <<- train_watchlist_split_val
    
    xgb_model <-
      xgboost_model_training_wrapper(
        input_data = model_data,
        response_var = response_var,
        id_vars = id_vars,
        replicate_unit_var = replicate_unit_var,
        features = features,
        response_type = response_type,
        train_test_split_val = train_test_split_val_x,
        train_watchlist_split_val = train_watchlist_split_val_x,
        
        balance_factor_by_smallest_class = T,
        
        print_arguments_bool = F)
    
    
    test_set_results <-
      evaluate_test_set.xboost.multiclass(
        xgboost_model = xgb_model$xgboost_model,
        test_data = xgb_model$xgb_data_list$test,
        test_xgb_matrix = xgb_model$model_matrices.xgb$test,
        response_var = response_var,
        id_vars = id_vars)
    
    
    test_set_summary <- test_set_results %>%
      summarise(mean_class_error = mean(class_error),
                mean_confidence = mean(predicted_class_probability)) %>%
      mutate(run = x)
    
    
    shap_values <- get_shap_values.xgboost(trained_xgb_model_object = xgb_model,
                                           response_var = response_var,
                                           predinteraction = F,
                                           problem_class = 'multiclass')
    
    shap_values.long <- process_shap_values.pivot_long(shap_values = shap_values)
    
    shap_summary <- summarise_shap_values.feature_importance(shap_values.long = shap_values.long) %>% arrange(-mean_absolute_SHAP)
    
    shap_summary.features_and_individuals <- full_join(shap_values.long, shap_summary) %>% arrange(-mean_absolute_SHAP)
    
    
    
    list(xgb_model = xgb_model,
         test_set_results = test_set_results,
         test_set_summary = test_set_summary,
         shap_summary.features_and_individuals = shap_summary.features_and_individuals)
    
  })
  
  
  test_set_metrics <- lapply(runs, function(x) { x$test_set_summary }) %>% bind_rows() %>%
    summarise(overall_mean_class_error = mean(mean_class_error),
              sd_class_error = sd(mean_class_error),
              max_class_error = max(mean_class_error),
              min_class_error = min(mean_class_error),
              overall_mean_confidence = mean(mean_confidence),
              sd_confidence = sd(mean_confidence))
  
  shap_summary.features_and_individuals <- lapply(runs, function(x) { 
    x$shap_summary.features_and_individuals 
  }) %>% bind_rows()
  
  
  out <- list(test_set_metrics = test_set_metrics,
              shap_summary.features_and_individuals = shap_summary.features_and_individuals)
  
  return(out)
  
}


generate_power_analysis_table <- function(sample_sizes, replicates) {
  
  out <- expand_grid(sample_size = sample_sizes, power_replicate = 1:replicates)
  
  return(out)
  
}


run_power_analysis <- function(model_data, 
                               sample_sizes,
                               replicates,
                               response_var,
                               response_type,
                               id_vars,
                               features,
                               replicate_unit_var,
                               train_test_split_val = 5,
                               train_watchlist_split_val = 0.2,
                               train_watchlist_split_type = 'proportion',
                               train_test_split_type = 'group_n',
                               static_test_set_bool = T) {
  
  power_analysis_table <- generate_power_analysis_table(sample_sizes = sample_sizes, 
                                                        replicates = replicates)
  
  if(static_test_set_bool) {
    
    print('power analysis running on static test set')
    
    model_data.power_analysis <- 
      set_up_model_data(
        input_data = model_data,
        response_var = response_var,
        features = features,
        id_vars = id_vars,
        replicate_unit_var = replicate_unit_var,
        train_test_split_type = train_test_split_type,
        response_type = response_type,
        train_test_split_val = train_test_split_val,
        train_watchlist_split_val = 0,
        split_function = split_data,
        
        smallest_class_n_override = NA
      )
    
    
    model_matrices.power_analysis.test <- 
      create_xgb_matrix(
        model_dataframe = model_data.power_analysis$test,
        response_var = response_var,
        response_type = response_type,
        features = features)
    
  } else {
    print('power analysis running on new test set for each iteration')
    
  }
  
  
  
  out <- lapply(1:nrow(power_analysis_table), function(x) {
    
    print(paste0('run ',x,' of ',nrow(power_analysis_table)))
    
    power_x <- power_analysis_table[x,]
    
    if(static_test_set_bool) {
      model_data_x <- model_data.power_analysis$train
    } else {
      model_data_x <- model_data
    }
    
    
    model_data_x <- model_data_x %>% 
      group_by(across(response_var)) %>%
      slice_sample(n = ifelse(static_test_set_bool, power_x$sample_size, power_x$sample_size + train_test_split_val)) %>%
      ungroup()
    
    
    xgb_model <- 
      xgboost_model_training_wrapper(
        input_data = model_data_x,
        response_var = response_var,
        replicate_unit_var = replicate_unit_var,
        id_vars = id_vars,
        features = features,
        response_type = response_type,
        
        train_test_split_type = train_test_split_type,
        train_test_split_val = ifelse(static_test_set_bool, 0, train_test_split_val),
        
        train_watchlist_split_val = train_watchlist_split_val,
        train_watchlist_split_type = train_watchlist_split_type,
        
        print_arguments_bool = F)
    
    
    test_set_results <- 
      evaluate_test_set.xboost.multiclass(
        xgboost_model = xgb_model$xgboost_model,
        
        test_data = if(static_test_set_bool) { model_data.power_analysis$test } else { xgb_model$xgb_data_list$test },
        
        test_xgb_matrix = if(static_test_set_bool) { model_matrices.power_analysis.test } else { xgb_model$model_matrices.xgb$test },
        
        response_var = response_var,
        id_vars = id_vars)
    
    test_set_summary <- test_set_results %>% summarise(mean_class_error = mean(class_error),
                                                       mean_confidence = mean(predicted_class_probability))
    
    out_x <- test_set_summary %>% bind_cols(power_x)
    
  }) %>% bind_rows() %>% mutate(static_test_set_bool = static_test_set_bool)
  
  return(out)
  
}


summarise_power_analysis <- function(power_analysis) {
  
  power_analysis_2 <- power_analysis %>%
    group_by(sample_size) %>%
    mutate(cumulative_mean_class_error = cummean(mean_class_error),
           cumulative_mean_confidence = cummean(mean_confidence))
  
  
  out <- power_analysis_2 %>% 
    group_by(sample_size) %>%
    summarise(sd_class_error = sd(mean_class_error),
              mean_class_error = mean(mean_class_error),
              sd_confidence = sd(mean_confidence),
              mean_confidence = mean(mean_confidence),
              reps = n(),
              static_test_set_bool = first(static_test_set_bool))
  
  return(out)
  
}


plot_power_analysis.raw <- function(power_analysis) {
  
  reps <- power_analysis$power_replicate %>% max()
  
  test_set_settings <- if(power_analysis$static_test_set_bool[1]) {
    'all runs evaluated against the same test set'
  } else {
    'all runs evaluated against a new test set each iteration'
  }
  
  
  p1 <- ggplot(power_analysis,
               aes(x = sample_size,
                   y = mean_class_error,
                   fill = mean_confidence)) +
    geom_jitter(size = 4, shape = 21, alpha = 0.7, width = 1, height = 0.00) +
    geom_smooth(colour = 'red', method = 'loess') +
    scale_y_continuous(limits = c(0,1)) +
    labs(title = 'The effect of sample size on mean class error',
         subtitle = paste0('Each sample size replicated ',reps,' times, ', test_set_settings)) +
    theme_minimal()
  
  p2 <- ggplot(power_analysis,
               aes(x = sample_size,
                   y = mean_confidence,
                   fill = mean_class_error)) +
    geom_jitter(size = 4, shape = 21, alpha = 0.7, width = 1, height = 0.00) +
    geom_smooth(colour = 'red', method = 'loess') +
    scale_y_continuous(limits = c(0,1)) +
    labs(title = 'The effect of sample size on mean confidence',
         subtitle = paste0('Each sample size replicated ',reps,' times, ', test_set_settings)) +
    theme_minimal()
  
  both <- grobTree(rectGrob(gp=gpar(fill="#FFFFFF", col="#FFFFFF", lwd=0)),  # background filling
                   grid.arrange(p1,                                        # plot 1 in layout matrix - main plot
                                p2,                                        # plot 2 in layout matrix - main plot
                                
                                # numbers refer to where the above plots will be placed
                                layout_matrix=rbind(c(1,1,1,1,1,1,2,2,2,2,2,2),
                                                    c(1,1,1,1,1,1,2,2,2,2,2,2),
                                                    c(1,1,1,1,1,1,2,2,2,2,2,2),
                                                    c(1,1,1,1,1,1,2,2,2,2,2,2),
                                                    c(1,1,1,1,1,1,2,2,2,2,2,2),
                                                    c(1,1,1,1,1,1,2,2,2,2,2,2))
                   )
  )
  
  out <- list(error = p1,
              confidence = p2,
              both = both)
  
  return(out)
  
}


plot_power_analysis.summary <- function(power_analysis_summarised) {
  
  reps <- power_analysis_summarised$reps[1]
  
  test_set_settings <- if(power_analysis_summarised$static_test_set_bool[1]) {
    'all runs evaluated against the same test set'
  } else {
    'all runs evaluated against a new test set each iteration'
  }
  
  
  p1 <- ggplot(power_analysis_summarised,
               aes(x = sample_size,
                   y = mean_class_error,
                   ymin = mean_class_error - sd_class_error,
                   ymax = mean_class_error + sd_class_error,
                   fill = mean_confidence)) +
    # geom_smooth(colour = 'red', method = 'loess') +
    geom_errorbar() +
    geom_point(size = 4, shape = 21, alpha = 1) +
    scale_y_continuous(limits = c(0,1)) +
    labs(title = 'The effect of sample size on mean class error',
         subtitle = paste0('Each sample size replicated ',reps,' times, ', test_set_settings)) +
    theme_minimal()
  
  p2 <- ggplot(power_analysis_summarised,
               aes(x = sample_size,
                   y = mean_confidence,
                   ymin = mean_confidence - sd_confidence,
                   ymax = mean_confidence + sd_confidence,
                   fill = mean_class_error)) +
    # geom_smooth(colour = 'red', method = 'loess') +
    geom_errorbar() +
    geom_point(size = 4, shape = 21, alpha = 1) +
    scale_y_continuous(limits = c(0,1)) +
    labs(title = 'The effect of sample size on mean confidence',
         subtitle = paste0('Each sample size replicated ',reps,' times, ', test_set_settings)) +
    theme_minimal()
  
  both <- grobTree(rectGrob(gp=gpar(fill="#FFFFFF", col="#FFFFFF", lwd=0)),  # background filling
                   grid.arrange(p1,                                        # plot 1 in layout matrix - main plot
                                p2,                                        # plot 2 in layout matrix - main plot
                                
                                # numbers refer to where the above plots will be placed
                                layout_matrix=rbind(c(1,1,1,1,1,1,2,2,2,2,2,2),
                                                    c(1,1,1,1,1,1,2,2,2,2,2,2),
                                                    c(1,1,1,1,1,1,2,2,2,2,2,2),
                                                    c(1,1,1,1,1,1,2,2,2,2,2,2),
                                                    c(1,1,1,1,1,1,2,2,2,2,2,2),
                                                    c(1,1,1,1,1,1,2,2,2,2,2,2))
                   )
  )
  
  out <- list(error = p1,
              confidence = p2,
              both = both)
  
  
  return(out)
  
}



# ..... Neural net library ====


generate_nnet_formula <- function(response_var, features) {
  
  r <- paste(response_var, collapse = '+')
  f <- paste(features, collapse = '+')
  
  out <- paste0(r,'~',f) %>% as.formula()
  
  return(out)
  
}


generate_nnet_layer_topology <- function(n_features, 
                                         hidden_layer_shape = c('none', 'funnel', 'log.funnel','block', 'bulge'),
                                         hidden_layer_depth = 5,
                                         node_count_scaling_factor = 1.1,
                                         layer_override = NA) {
  
  if(!is.na(layer_override)) {
    
    return(layer_override)
    
  }
  
  if(hidden_layer_shape == 'none') {
    
    return(0)
    
  } else {
    
    init <- rep_len(1, length.out = hidden_layer_depth) %>% 
      as_tibble() %>% 
      mutate(pos = row_number()) %>% 
      arrange(-pos) %>%
      mutate(pos_2 = row_number()) %>%
      mutate(value_funnel = ((value * pos) / hidden_layer_depth) * (n_features * node_count_scaling_factor),
             value_log.funnel = ((value * exp(pos)) / hidden_layer_depth) * (n_features * node_count_scaling_factor),
             value_log.funnel = value_log.funnel / max(value_log.funnel) * (n_features * node_count_scaling_factor),
             value_block = n_features * node_count_scaling_factor,
             value_bulge = pos^pos_2,
             value_bulge = value_bulge / max(value_bulge) * (n_features * node_count_scaling_factor)) %>%
      mutate(across(matches('value'), ceiling))
    
  }
  
  out <- init %>% dplyr::select(matches(paste0('_',hidden_layer_shape[1]))) %>% .[[1]]
  
  return(out)
  
}


evaluate_test_set.nnet <- function(nnet_model, 
                                   target_data, 
                                   response_var,
                                   id_vars,
                                   response_type = c('regression', 'classification'),
                                   include_geodesic = T) {
  
  if(response_type[1] == 'classification') {
    
    fitted <- predict(object = nnet_model, newdata = target_data) %>% as_tibble()
    names(fitted) <- unique(arrange(target_data[response_var]))[[response_var]] %>% as.character()
    
    predict_data <- target_data %>% 
      dplyr::select(all_of(c(response_var, id_vars))) %>%
      bind_cols(fitted)
    
    predict_data <- probabilities_to_classes(probability_table = predict_data,
                                             response_var = response_var,
                                             id_vars = id_vars)
    
    
    predict_data_summary <- predict_data %>% summarise(mean_class_error = mean(class_error))
    
  } else {
    
    fitted <- predict(object = nnet_model, newdata = target_data) %>% as_tibble() %>% mutate(across(matches('V'), ~round(x = .x, digits = 3)))
    names(fitted) <- paste0(response_var,'_fitted')
    if(include_geodesic) {
      
      geodesic_error_distance <- geodist(x = target_data %>% dplyr::select(response_var),
                                         y = fitted,
                                         paired = TRUE,
                                         measure =  "geodesic")
      
    } else {
      
      geodesic_error_distance <- NA
      
    }
    
    
    predict_data <- target_data %>% 
      dplyr::select(all_of(c(id_vars, response_var))) %>%
      bind_cols(fitted) %>%
      mutate(latitude_error = latitude_fitted - latitude,
             latitude_error_abs = abs(latitude_error),
             longitude_error = longitude_fitted - longitude,
             longitude_error_abs = abs(longitude_error),
             geodesic_error_distance_km = ifelse(include_geodesic, round(geodesic_error_distance/1000, 0), NA)
      )
    
    if(include_geodesic == F) predict_data <- predict_data %>% dplyr::select(-geodesic_error_distance_km)
    
    predict_data_summary <- predict_data %>% summarise(across(matches('error'), list(mean = mean, sd = sd)))
    
  }
  
  out <- list(predict_data = predict_data,
              predict_data_summary = predict_data_summary)
  
  return(out)
  
}
