# Model plotting


# Plots ====


# ..... Model performance summary - Scatterplot
if(p.scatter) {
  
  plot <- ggplot(data = xgb_models.predictions,
                 aes(x = get(predict_var), y = get(actual_var))) +
    
    geom_point() +
    
    geom_abline(slope = 1, intercept = 0, size = 2, colour = 'red') +
    
    theme_minimal(base_size = 22) + 
    theme(
      #text = element_text(colour = '#00632e'), 
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
  
  ggsave(filename = paste0(out_dir,'model_performance_tree_booster_',response_var_out,'_scatter.png'),
         plot = plot,
         width = 1200,
         height = 1080,
         scale = 3,
         units = 'px')
  
  saveRDS(plot,
          file = paste0(out_dir,'model_performance_tree_booster_',response_var_out,'_scatter.rds'))
  
  
}



# ..... Model performance summary - boxplot ====

if(p.boxplot) {
  
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
         units = 'px')
  
  saveRDS(object = plot,
          file = paste0(out_dir,'model_performance_tree_booster_',response_var_out,'_boxplot.rds'))
  
}


# .....  Feature importance ====
if(p.shaps) {
  
  shap_plot <- ggplot(data = shap_values.long %>% filter(data_type == 'train'),
                      aes(y = shap_abs, x = reorder(feature, shap_abs))) +
    
    geom_hline(yintercept = 0) +
    
    geom_bar(fill = '#008822', alpha = 0.6, size = 1.5,
             stat = 'summary', fun = 'sum') +
    
    scale_y_continuous(expand = c(0, 0)) + 
    
    coord_flip() +
    
    labs(x = 'feature') +
    
    theme_bw() +
    theme(# axis.text.x = element_blank(),
      # axis.ticks.x = element_blank(),
      # axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
      
      # axis.text.y = element_blank(),
      # axis.ticks.y = element_blank(),
      panel.grid = element_blank())
  
  shap_plot
  
  ggsave(filename = paste0(out_dir,'feature_importance_shap_values_',response_var_out,'.png'),
         plot = shap_plot,
         width = 1900,
         height = 1080,
         scale = 1,
         units = 'px')
  
  saveRDS(object = plot,
          file = paste0(out_dir,'feature_importance_shap_values_',response_var_out,'.rds'))
  
}


# ..... Map ====

# prediction_summary <- xgb_models.predictions %>%
#   group_by(across(id_vars)) %>%
#   summarise(mean_prediction = mean(get(predict_var)),
#             sd_prediction = sd(get(predict_var)),
#             mean_abs_error = mean(error.abs),
#             sd_abs_error = sd(error.abs)) %>%
#   pivot_longer(cols = c('mean_prediction', 
#                         'sd_prediction',
#                         'mean_abs_error',
#                         'sd_abs_error'),
#                names_to = 'variable',
#                values_to = 'value')
# 
# plot <- ggplot(data = prediction_summary %>% filter(variable == 'mean_abs_error'),
#                aes(x = pu_lon, y = pu_lat, colour = value)) +
#   
#   geom_point(alpha = 0.3) +
#   
#   facet_wrap(~variable) +
#   
#   scale_color_viridis_c() +
#   
#   theme_bw()
# 
# plot


# .....  PU predictions ====

if(p.pu) {

  plot <- ggplot(data = pu_preds.aggregated %>% filter(country == 'kenya'),
                 mapping = aes(x = pu_lon, 
                               y = pu_lat,
                               colour = get(paste0(predict_var,'.mean')))) +

    geom_point(size = 2, alpha = 0.6) +
    
    coord_equal() +
    
    labs(colour = paste0('Mean predicted\n',response_var_out,'\nXGBoost')) +
    
    theme_bw() +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank())
  
  plot
  
  
  ggsave(filename = paste0(out_dir,'planning_units_predictions_kenya_',response_var_out,'.png'),
         plot = plot,
         width = 1200,
         height = 1080,
         scale = 3,
         units = 'px')
  
  saveRDS(object = plot,
          file = paste0(out_dir,'planning_units_predictions_kenya_',response_var_out,'.rds'))
  
}


# ..... PU predictions - kenya ====

if(p.pu_kenya) {

  plot <- ggplot(data = pu_preds.aggregated %>% filter(country == 'kenya'),
                 mapping = aes(x = pu_lon, 
                               y = pu_lat,
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
  
  plot
  
  
  ggsave(filename = paste0(out_dir,'planning_units_prediction_uncertainty_kenya_',response_var_out,'.png'),
         plot = plot,
         width = 1200,
         height = 1080,
         scale = 3,
         units = 'px')
  
  saveRDS(object = plot,
          file = paste0(out_dir,'planning_units_prediction_uncertainty_kenya_',response_var_out,'.rds'))
  
}


# ..... Model error by country ====
if(p.pu_error_by_country) {
  
  plot <- ggplot(data = pu_preds.aggregated,
                 mapping = aes(x = pu_lon, 
                               y = pu_lat,
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
  
  plot
  
  
  ggsave(filename = paste0(out_dir,'planning_units_model_error_by_country_',response_var_out,'.png'),
         plot = plot,
         width = 1200,
         height = 1080,
         scale = 3,
         units = 'px')
  
  saveRDS(object = plot,
          file = paste0(out_dir,'planning_units_model_error_by_country_',response_var_out,'.rds'))
  
  
}


# .....  XGBoost Vs gam - difference map ====
if(p.xgb_vs_gam) {
  
  plot <- ggplot(data = pu_preds.aggregated,
                 mapping = aes(x = pu_lon, 
                               y = pu_lat,
                               colour = xgb_vs_gam_diff)) +
    
    geom_point(size = 1, alpha = 0.6) +
    
    coord_equal() +
    
    # scale_color_gradient(low = "#CCCC00",
    #                      high = "#CC0000") +
    
    scale_colour_gradient2(low = '#FF2222', 
                           mid = '#cccccc',
                           high = '#2222FF') +
    
    labs(colour = 'XGBoost Vs gam\ndifference') +
    
    theme_bw() +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          panel.grid = element_blank())
  
  plot
  
  
  ggsave(filename = paste0(out_dir,'planning_units_model_xgb_vs_gam_diff_',response_var_out,'.png'),
         plot = plot,
         width = 1200,
         height = 1080,
         scale = 3,
         units = 'px')
  
  saveRDS(object = plot,
          file = paste0(out_dir,'planning_units_model_xgb_vs_gam_diff_',response_var_out,'.rds'))
  
}


# .....  XGBoost Vs gam - difference scatter ====

if(p.xgb_vs_gam_scatter) {

  plot <- ggplot(pu_preds.aggregated, 
                 aes(x = Coralcover.prop.predicted.gam,
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
  
  plot
  
  
  ggsave(filename = paste0(out_dir,'planning_units_model_xgb_vs_gam_diff_scatter_',response_var_out,'.png'),
         plot = plot,
         width = 1080,
         height = 1080,
         scale = 3,
         units = 'px')
  
  saveRDS(object = plot,
          file = paste0(out_dir,'planning_units_model_xgb_vs_gam_diff_scatter_',response_var_out,'.rds'))
  
}


# .....  PU predictions - gam ====
if(p.pu_preds_gam) {
  
  plot <- ggplot(data = pu_preds.aggregated,
                 mapping = aes(x = pu_lon, 
                               y = pu_lat,
                               colour = get(paste0(predict_var,'.gam')))) +
    
    geom_point(size = 2, alpha = 0.6) +
    
    coord_equal() +
    
    labs(colour = 'Mean predicted\nmangrove cover\ngam') +
    
    theme_bw() +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank())
  
  plot
  
  
  ggsave(filename = paste0(out_dir,'planning_units_predictions_gam_',response_var_out,'.png'),
         plot = plot,
         width = 1200,
         height = 1080,
         scale = 3,
         units = 'px')
  
  saveRDS(object = plot,
          file = paste0(out_dir,'planning_units_predictions_gam_',response_var_out,'.rds'))
  
}


# .....  PU predictions - gam - kenya ====
if(p.pu_preds_gam_kenya) {

  plot <- ggplot(data = pu_preds.aggregated %>% filter(country == 'kenya'),
                 mapping = aes(x = pu_lon, 
                               y = pu_lat,
                               colour = get(paste0(predict_var,'.gam')))) +
    
    geom_point(size = 2, alpha = 0.6) +
    
    coord_equal() +
    
    labs(colour = 'Mean predicted\nmangrove cover\ngam') +
    
    theme_bw() +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank())
  
  plot
  
  
  ggsave(filename = paste0(out_dir,'planning_units_predictions_gam_kenya_',response_var_out,'.png'),
         plot = plot,
         width = 1200,
         height = 1080,
         scale = 3,
         units = 'px')
  
  saveRDS(object = plot,
          file = paste0(out_dir,'planning_units_predictions_gam_kenya_',response_var_out,'.rds'))
  
  
}


# .....  XGBoost Vs gam - difference map - kenya ====
if(p.xgb_vs_gam_map) {
  
  plot <- ggplot(data = pu_preds.aggregated %>% filter(country == 'kenya'),
                 mapping = aes(x = pu_lon, 
                               y = pu_lat,
                               colour = xgb_vs_gam_diff)) +
    
    geom_point(size = 2, alpha = 0.6) +
    
    coord_equal() +
    
    # scale_color_gradient(low = "#CCCC00",
    #                      high = "#CC0000") +
    
    scale_colour_gradient2(low = '#FF2222', 
                           mid = '#cccccc',
                           high = '#2222FF') +
    
    labs(colour = 'XGBoost Vs gam\ndifference') +
    
    theme_bw() +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          panel.grid = element_blank())
  
  plot
  
  
  ggsave(filename = paste0(out_dir,'planning_units_model_xgb_vs_gam_diff_kenya_',response_var_out,'.png'),
         plot = plot,
         width = 1200,
         height = 1080,
         scale = 3,
         units = 'px')
  
  saveRDS(object = plot,
          file = paste0(out_dir,'planning_units_model_xgb_vs_gam_diff_kenya_',response_var_out,'.png'))
  
}

