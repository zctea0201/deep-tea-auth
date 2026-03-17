# Load necessary libraries
library(tidyverse)
library(furrr)

# ==========================================
# 1. Data Augmentation Helper Function
# ==========================================
data_augmentation <- function(
    data, 
    rt_mean = 10, rt_sd = 2, 
    mz_mean = 10, mz_sd = 2, 
    intensity_mean = 0.1, intensity_sd = 0.1,
    rt_min, rt_max, mz_min, mz_max # Added these to ensure filtering uses correct scope
) {
  
  # Function to add random noise to the RT values
  get_rt_shift <- function(rt, rt_mean, rt_sd) {
    error <- rnorm(n = 10000, mean = rt_mean, sd = rt_sd)
    rt <- purrr::map(rt, .f = function(x) {
      temp_error <- sample(error, 1)
      return(x + temp_error)
    })
    unname(unlist(rt))
  }
  
  # Function to add random noise to the m/z values
  get_mz_shift <- function(mz, mz_mean, mz_sd) {
    error <- rnorm(n = 10000, mean = mz_mean, sd = mz_sd)
    mz <- purrr::map(mz, .f = function(x) {
      temp_error <- sample(error, 1)
      # Note: Preserving original logic for mz shift calculation
      return(x - (ifelse(x < 400, 400, x) * temp_error) / 10^6)
    })
    unname(unlist(mz))
  }
  
  # Function to add random noise to the intensity values
  get_intensity_shift <- function(intensity, intensity_mean, intensity_sd) {
    error <- rnorm(n = 10000, mean = intensity_mean, sd = intensity_sd)
    intensity <- purrr::map(intensity, .f = function(x) {
      temp_error <- sample(error, 1)
      return(x * (1 + temp_error))
    })
    unname(unlist(intensity))
  }
  
  # Apply intensity shift to the data frame columns
  apply_intensity_shift_to_df <- function(df, intensity_mean, intensity_sd) {
    sample_cols <- names(df)[3:ncol(df)]
    df_shifted <- df %>%
      mutate(across(all_of(sample_cols), 
                    ~get_intensity_shift(., intensity_mean = intensity_mean, intensity_sd = intensity_sd)))
    return(df_shifted)
  }
  
  # Apply the random noise
  # Ensure we work on a copy to avoid reference issues
  data_mod <- data 
  data_mod$rt <- get_rt_shift(data_mod$rt, rt_mean = rt_mean, rt_sd = rt_sd)
  data_mod$mz <- get_mz_shift(data_mod$mz, mz_mean = mz_mean, mz_sd = mz_sd)
  data_mod <- apply_intensity_shift_to_df(data_mod, intensity_mean = intensity_mean, intensity_sd = intensity_sd)
  
  # Filter based on ranges
  data_shifted <- data_mod %>%
    filter(rt >= rt_min, rt <= rt_max, mz >= mz_min, mz <= mz_max)
  
  return(data_shifted)
}
