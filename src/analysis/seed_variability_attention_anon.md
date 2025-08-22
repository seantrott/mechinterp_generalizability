---
title: "Variation in previous-token attention"
author: "Anonymous Authors"
date: "August 15, 2025"
output:
  html_document:
    keep_md: yes
    toc: yes
    toc_float: yes
---






# Load Pythia data


Here, we analyze *summary data* looking at the average attention each head gives from each token to the previous token. 




# Final-step attention

## Previous token heads: a summary


``` r
df_by_head_seed = df_pythia_models %>%
  filter(step_modded == 143001) %>%
  group_by(mpath) %>%
  mutate(z_mean_prev_self_ratio = scale(mean_prev_self_ratio))


### Max by layer
df_by_head_seed %>%
  group_by(mpath, seed_name, Layer, n_params) %>%
  summarise(max_mean = max(mean_prev_self_ratio)) %>%
  ggplot(aes(x = Layer,
             y = seed_name,
             fill = max_mean)) +
  geom_tile() +
  labs(x = "Layer",
       y = "Seed",
       fill = "Max Previous/Self Attn. Ratio") +
  scale_fill_gradient2(low = "blue",
                       mid = "white",
                       high = "red",
                       midpoint = 0, 
                       name = "Max Previous/Self Attn. Ratio") +
  theme_minimal() +
  theme(text = element_text(size = 15),
        strip.text.y = element_text(angle = 0), # Keep facet labels horizontal
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom") +
  facet_wrap(~reorder(mpath, n_params))
```

```
## `summarise()` has grouped output by 'mpath', 'seed_name', 'Layer'. You can
## override using the `.groups` argument.
```

![](seed_variability_attention_anon_files/figure-html/final_step_1back-1.pdf)<!-- -->

``` r
### Max by layer
df_by_head_seed %>%
  group_by(mpath, seed_name, Layer, n_params) %>%
  summarise(max_mean = max(z_mean_prev_self_ratio)) %>%
  ggplot(aes(x = Layer,
             y = seed_name,
             fill = max_mean)) +
  geom_tile() +
  labs(x = "Layer",
       y = "Seed",
       fill = "Max Previous/Self Attn. Ratio (z-scored)") +
  scale_fill_gradient2(low = "blue",
                       mid = "white",
                       high = "red",
                       midpoint = 0, 
                       name = "Max Previous/Self Attn. Ratio (z-scored)") +
  theme_minimal() +
  theme(text = element_text(size = 15),
        strip.text.y = element_text(angle = 0), # Keep facet labels horizontal
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom") +
  facet_wrap(~reorder(mpath, n_params))
```

```
## `summarise()` has grouped output by 'mpath', 'seed_name', 'Layer'. You can
## override using the `.groups` argument.
```

![](seed_variability_attention_anon_files/figure-html/final_step_1back-2.pdf)<!-- -->

``` r
### How does *max* attention per layer change across model size?
df_layerwise_attn <- df_by_head_seed %>%
  group_by(mpath, Layer, n_params, seed_name) %>%
  summarise(
    max_attention = max(z_mean_prev_self_ratio),
    se_attention = sd(z_mean_prev_self_ratio) / sqrt(n()),
    .groups = "drop"
  )

### 
summary(lmer(data = df_layerwise_attn,
             max_attention ~ log10(n_params) * Layer + (1|seed_name)))
```

```
## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
## lmerModLmerTest]
## Formula: max_attention ~ log10(n_params) * Layer + (1 | seed_name)
##    Data: df_layerwise_attn
## 
## REML criterion at convergence: 1814.5
## 
## Scaled residuals: 
##     Min      1Q  Median      3Q     Max 
## -1.3782 -0.6761 -0.3052  0.3862  5.1249 
## 
## Random effects:
##  Groups    Name        Variance Std.Dev.
##  seed_name (Intercept) 0.02236  0.1495  
##  Residual              3.79778  1.9488  
## Number of obs: 432, groups:  seed_name, 9
## 
## Fixed effects:
##                        Estimate Std. Error        df t value Pr(>|t|)    
## (Intercept)           -11.02921    2.99027 421.02594  -3.688 0.000255 ***
## log10(n_params)         1.52220    0.36003 419.99996   4.228  2.9e-05 ***
## Layer                   1.98264    0.62270 419.99990   3.184 0.001561 ** 
## log10(n_params):Layer  -0.23085    0.07269 419.99990  -3.176 0.001604 ** 
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Correlation of Fixed Effects:
##             (Intr) lg10(_) Layer 
## lg10(n_prm) -0.998               
## Layer       -0.790  0.766        
## lg10(n_p):L  0.798 -0.775  -1.000
```


## Layer depth ratio

Where do these heads pop up?


``` r
df_pythia_models_final_step = df_pythia_models %>%
  filter(step_modded == 143001) %>%
  group_by(model) %>%
  mutate(max_layer = max(Layer),
         prop_layer = Layer / max_layer) %>%
  ### Scale for interpreting the coefficients more easily
  mutate(prop_layer_scaled = scale(prop_layer)) %>%
  ungroup()


# Summarize mean and SE by model and layer
summary_df <- df_pythia_models_final_step %>%
  group_by(model, n_params, seed_name, Layer) %>%
  summarise(
    max_ratio_per_seed = max(mean_prev_self_ratio, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(model, Layer, n_params) %>%
  summarise(
    avg_max_ratio = mean(max_ratio_per_seed, na.rm = TRUE),
    se_max_ratio = sd(max_ratio_per_seed, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

ggplot(summary_df, aes(x = Layer, y = avg_max_ratio, 
                       color = reorder(model, n_params))) +
  geom_line(size = 1.2) +
  geom_ribbon(aes(ymin = avg_max_ratio - se_max_ratio,
                  ymax = avg_max_ratio + se_max_ratio,
                  fill = reorder(model, n_params)), alpha = 0.2, color = NA) +
  labs(x = "Layer",
       y = "Max Previous/Self Attn. Ratio",
       color = "Model", fill = "Model") +
  scale_color_manual(values = viridisLite::viridis(4, option = "mako", 
                                                   begin = 0.8, end = 0.15)) + 
  scale_fill_manual(values = viridisLite::viridis(4, option = "mako", 
                                                   begin = 0.8, end = 0.15)) + 
  theme_minimal(base_size = 15) +
  theme(text = element_text(size = 15),
        legend.position = "bottom") 
```

```
## Warning: Using `size` aesthetic for lines was deprecated in ggplot2 3.4.0.
## ℹ Please use `linewidth` instead.
## This warning is displayed once every 8 hours.
## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
## generated.
```

![](seed_variability_attention_anon_files/figure-html/unnamed-chunk-3-1.pdf)<!-- -->

``` r
### Visualizing relative layer
summary_df <- df_pythia_models_final_step %>%
  mutate(binned_prop_layer = ntile(prop_layer, 6)) %>%
  mutate(prop_binned = binned_prop_layer / 6) %>%
  group_by(model, n_params, seed_name, prop_binned) %>%
  summarise(
    max_ratio_per_seed = max(mean_prev_self_ratio, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(model, prop_binned, n_params) %>%
  summarise(
    avg_max_ratio = mean(max_ratio_per_seed, na.rm = TRUE),
    se_max_ratio = sd(max_ratio_per_seed, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

ggplot(summary_df, aes(x = prop_binned, y = avg_max_ratio, 
                       color = reorder(model, n_params))) +
  geom_line(size = 1.2) +
  geom_ribbon(aes(ymin = avg_max_ratio - se_max_ratio,
                  ymax = avg_max_ratio + se_max_ratio,
                  fill = reorder(model, n_params)), alpha = 0.2, color = NA) +
  labs(x = "Layer Depth",
       y = "Max Previous/Self Attn. Ratio",
       color = "", fill = "") +
  scale_color_manual(values = viridisLite::viridis(4, option = "mako", 
                                                   begin = 0.8, end = 0.15)) + 
  scale_fill_manual(values = viridisLite::viridis(4, option = "mako", 
                                                   begin = 0.8, end = 0.15)) + 
  theme_minimal(base_size = 15) +
  theme(text = element_text(size = 15),
        legend.position = "bottom") 
```

![](seed_variability_attention_anon_files/figure-html/unnamed-chunk-3-2.pdf)<!-- -->

``` r
### Analyses
# Fit GAM across all models/seeds
summary(lmer(mean_prev_self_ratio ~ prop_layer + (1|Head) + (1|model), 
             data = df_pythia_models_final_step))
```

```
## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
## lmerModLmerTest]
## Formula: mean_prev_self_ratio ~ prop_layer + (1 | Head) + (1 | model)
##    Data: df_pythia_models_final_step
## 
## REML criterion at convergence: 10352.9
## 
## Scaled residuals: 
##     Min      1Q  Median      3Q     Max 
## -1.8134 -0.4803 -0.2319  0.1636 14.3924 
## 
## Random effects:
##  Groups   Name        Variance  Std.Dev.
##  Head     (Intercept) 0.0001647 0.01283 
##  model    (Intercept) 0.0181619 0.13477 
##  Residual             0.3965398 0.62971 
## Number of obs: 5400, groups:  Head, 16; model, 4
## 
## Fixed effects:
##              Estimate Std. Error        df t value Pr(>|t|)    
## (Intercept) 8.645e-01  7.087e-02 3.385e+00  12.199 0.000647 ***
## prop_layer  2.263e-01  2.977e-02 5.384e+03   7.602 3.41e-14 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Correlation of Fixed Effects:
##            (Intr)
## prop_layer -0.234
```

``` r
library(mgcv)
```

```
## Loading required package: nlme
```

```
## 
## Attaching package: 'nlme'
```

```
## The following object is masked from 'package:lme4':
## 
##     lmList
```

```
## The following object is masked from 'package:dplyr':
## 
##     collapse
```

```
## This is mgcv 1.9-1. For overview type 'help("mgcv-package")'.
```

``` r
# Fit GAM across all models/seeds
gam_all <- gam(mean_prev_self_ratio ~ s(prop_layer), data = df_pythia_models_final_step)

# Summary of the GAM
summary(gam_all)
```

```
## 
## Family: gaussian 
## Link function: identity 
## 
## Formula:
## mean_prev_self_ratio ~ s(prop_layer)
## 
## Parametric coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept) 0.936588   0.008512     110   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Approximate significance of smooth terms:
##                 edf Ref.df     F p-value    
## s(prop_layer) 8.259  8.847 35.59  <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## R-sq.(adj) =  0.0545   Deviance explained = 5.59%
## GCV = 0.39197  Scale est. = 0.3913    n = 5400
```

``` r
# Plot the GAM curve
plot(gam_all)
```

![](seed_variability_attention_anon_files/figure-html/unnamed-chunk-3-3.pdf)<!-- -->

``` r
# Create a prediction dataframe
pred_df <- df_pythia_models_final_step %>%
  select(prop_layer) %>%
  distinct() %>%
  arrange(prop_layer) %>%
  mutate(gam_pred = predict(gam_all, newdata = .))

# Overlay GAM on summary_df plot
ggplot(summary_df, aes(x = prop_binned, y = avg_max_ratio, 
                       color = reorder(model, n_params), fill = reorder(model, n_params))) +
  geom_line(size = 1.2) +
  geom_ribbon(aes(ymin = avg_max_ratio - se_max_ratio,
                  ymax = avg_max_ratio + se_max_ratio), alpha = 0.2, color = NA) +
  geom_line(data = pred_df, aes(x = prop_layer, y = gam_pred),
            inherit.aes = FALSE, color = "black", size = 1, linetype = "dashed") +
  labs(x = "Layer Depth",
       y = "Max Previous/Self Attn. Ratio",
       color = "", fill = "") +
  scale_color_manual(values = viridisLite::viridis(4, option = "mako", begin = 0.8, end = 0.15)) +
  scale_fill_manual(values = viridisLite::viridis(4, option = "mako", begin = 0.8, end = 0.15)) +
  theme_minimal(base_size = 15) +
  theme(text = element_text(size = 15),
        legend.position = "bottom")
```

![](seed_variability_attention_anon_files/figure-html/unnamed-chunk-3-4.pdf)<!-- -->



# Attention over time

## Previous token heads


``` r
### Track max previous token heads at each time point for each seed, across layer/head
df_by_head_max_attention = df_pythia_models %>%
  group_by(model, n_params, step_modded, seed, seed_name) %>%
  slice_max(mean_prev_self_ratio)


summary_avg = df_by_head_max_attention %>%
  group_by(model, n_params, step_modded) %>%
  summarise(
    mean_across_seeds = mean(mean_prev_self_ratio)
  )
```

```
## `summarise()` has grouped output by 'model', 'n_params'. You can override using
## the `.groups` argument.
```

``` r
df_by_head_max_attention %>%
  ggplot(aes(x = step_modded,
             y = mean_prev_self_ratio,
             color = factor(seed_name))) +
  geom_line(size = .6) +  # Lineplot for mean entropy
  geom_line(data = summary_avg, aes(x = step_modded, y = mean_across_seeds), 
             color = "black", size = 1.5) + # Smoothed average 
  theme_minimal() +
  labs(x = "Training Step (Log10)",
       y = "Max Previous/Self Attention Ratio",
       color = "") +
  scale_x_log10() +
  geom_vline(xintercept = 1000, 
           linetype = "dotted", 
           size = 1.2) +
  theme(text = element_text(size = 15),
        legend.position = "bottom") +
  scale_color_manual(values = viridisLite::viridis(9, option = "mako", 
                                                   begin = 0.8, end = 0.15)) + 
  facet_wrap(~reorder(model, n_params))
```

![](seed_variability_attention_anon_files/figure-html/previous_token_over_time-1.pdf)<!-- -->

``` r
summary(lmer(data = df_by_head_max_attention,
             mean_prev_self_ratio ~ log10(step_modded) + (1|seed_name) + (1|model),
             REML = FALSE))
```

```
## boundary (singular) fit: see help('isSingular')
```

```
## Linear mixed model fit by maximum likelihood . t-tests use Satterthwaite's
##   method [lmerModLmerTest]
## Formula: mean_prev_self_ratio ~ log10(step_modded) + (1 | seed_name) +  
##     (1 | model)
##    Data: df_by_head_max_attention
## 
##       AIC       BIC    logLik -2*log(L)  df.resid 
##    1843.7    1865.5    -916.8    1833.7       571 
## 
## Scaled residuals: 
##     Min      1Q  Median      3Q     Max 
## -2.2153 -0.6727 -0.1336  0.5233  4.9861 
## 
## Random effects:
##  Groups    Name        Variance Std.Dev.
##  seed_name (Intercept) 0.0000   0.000   
##  model     (Intercept) 0.1354   0.368   
##  Residual              1.3865   1.177   
## Number of obs: 576, groups:  seed_name, 9; model, 4
## 
## Fixed effects:
##                     Estimate Std. Error        df t value Pr(>|t|)    
## (Intercept)         -0.08196    0.20185   5.04817  -0.406    0.701    
## log10(step_modded)   1.01014    0.02967 572.00000  34.041   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Correlation of Fixed Effects:
##             (Intr)
## lg10(stp_m) -0.332
## optimizer (nloptwrap) convergence code: 0 (OK)
## boundary (singular) fit: see help('isSingular')
```

## Individual heads


``` r
### 14M
df_pythia_models %>%
  filter(Layer == 3) %>%
  filter(model == "pythia-14m") %>%
  ggplot(aes(x = step_modded,
             y = mean_prev_self_ratio,
             color = factor(Head))) +
  geom_point(size = 1.5, alpha = .7) +
  geom_line(size = 2, alpha = .7) +
  theme_minimal() +
  labs(x = "Training Step (Log10)",
       y = "Previous/Self Attention Ratio",
       color = "Attention Head",
       title = "Layer 3 Heads (14M)") +
  scale_x_log10() +
  geom_vline(xintercept = 1000, 
           linetype = "dotted", 
           size = 1.2) +
  theme(text = element_text(size = 15),
        legend.position = "bottom") +
  scale_color_viridis(option = "mako", discrete=TRUE) +
  facet_wrap(~seed_name)
```

![](seed_variability_attention_anon_files/figure-html/individual_heads-1.pdf)<!-- -->

``` r
### 14M
df_pythia_models %>%
  filter(Layer == 4) %>%
  filter(model == "pythia-14m") %>%
  ggplot(aes(x = step_modded,
             y = mean_prev_self_ratio,
             color = factor(Head))) +
  geom_point(size = 1.5, alpha = .7) +
  geom_line(size = 2, alpha = .7) +
  theme_minimal() +
  labs(x = "Training Step (Log10)",
       y = "Previous/Self Attention Ratio",
       color = "Attention Head",
       title = "Layer 4 Heads (14M)") +
  scale_x_log10() +
  geom_vline(xintercept = 1000, 
           linetype = "dotted", 
           size = 1.2) +
  theme(text = element_text(size = 15),
        legend.position = "bottom") +
  scale_color_viridis(option = "mako", discrete=TRUE) +
  facet_wrap(~seed_name)
```

![](seed_variability_attention_anon_files/figure-html/individual_heads-2.pdf)<!-- -->


## Research questions

Do bigger models show an earlier onset of their biggest change?


``` r
### First, try looking at first onset of a big change, trying out different ratios
### Calculate cross-step ratio
df_diff <- df_by_head_max_attention %>%
  group_by(mpath, seed_name) %>%
  arrange(step_modded) %>%
  mutate(
    log_step = log10(step_modded),
    d_ratio = c(NA, diff(mean_prev_self_ratio) / diff(log_step))
  ) %>%
  ungroup()

# Define a range of d_ratio thresholds to test
d_ratio_thresholds <- seq(0.1, 1.5, by = 0.1)

# Initialize a list to store the results for each d_ratio threshold
results <- list()

# Iterate over d_ratio thresholds
for (threshold in d_ratio_thresholds) {
  # Filter and find the emergence onset for each threshold
  emergence_onset <- df_diff %>%
    filter(d_ratio > threshold) %>%
    group_by(model, seed_name, n_params) %>%  # Group by model size (n_params) and seed_name
    slice_min(step_modded, with_ties = FALSE) %>%
    ungroup()

  # Store the emergence step for each threshold, seed, and model size
  emergence_step_data <- emergence_onset %>%
    select(model, seed_name, n_params, step_modded, d_ratio) %>%
    mutate(d_ratio_threshold = threshold)  # Add the threshold as a column

  # Append the result for this threshold to the list
  results[[as.character(threshold)]] <- emergence_step_data
}

# Combine all the results into one data frame
results_df <- bind_rows(results)

results_summ = results_df %>%
  group_by(model, n_params, seed_name) %>%
  summarise(m_step = mean(step_modded - 1)) 
```

```
## `summarise()` has grouped output by 'model', 'n_params'. You can override using
## the `.groups` argument.
```

``` r
### Plot average across d_ratio
results_summ %>%
  ggplot(aes(x = n_params,
             y = m_step,
             color = model)) +
  geom_jitter(size = 5, alpha = .7, width = .05) +
  scale_color_manual(values = viridisLite::viridis(4, option = "mako", 
                                                   begin = 0.8, end = 0.15)) + 
  scale_x_log10() +
  scale_y_log10() +
  theme_minimal() +
  theme(text = element_text(size = 15),
        legend.position = "bottom") +
  labs(x = "Number of Parameters (Log10)",
       y = "Max. Deriv. Step (Log10)",
       color = "") 
```

![](seed_variability_attention_anon_files/figure-html/onset-1.pdf)<!-- -->

``` r
### For each order magnitude in model size, how much sooner do we expect first onset on avearge?
mod_ratio = lmer(data = results_summ,
                 log10(m_step) ~ log10(n_params) + (1|seed_name))
```

```
## boundary (singular) fit: see help('isSingular')
```

``` r
summary(mod_ratio)
```

```
## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
## lmerModLmerTest]
## Formula: log10(m_step) ~ log10(n_params) + (1 | seed_name)
##    Data: results_summ
## 
## REML criterion at convergence: 18.9
## 
## Scaled residuals: 
##      Min       1Q   Median       3Q      Max 
## -2.18345 -0.52812  0.08928  0.78478  1.46839 
## 
## Random effects:
##  Groups    Name        Variance Std.Dev.
##  seed_name (Intercept) 0.00000  0.0000  
##  Residual              0.08577  0.2929  
## Number of obs: 36, groups:  seed_name, 9
## 
## Fixed effects:
##                 Estimate Std. Error       df t value Pr(>|t|)    
## (Intercept)      9.61868    0.72451 34.00000  13.276 5.22e-15 ***
## log10(n_params) -0.84852    0.09089 34.00000  -9.336 6.57e-11 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Correlation of Fixed Effects:
##             (Intr)
## lg10(n_prm) -0.998
## optimizer (nloptwrap) convergence code: 0 (OK)
## boundary (singular) fit: see help('isSingular')
```

``` r
#### Now identify inflection points with GAMs
library(mgcv)
library(dplyr)
library(ggplot2)

# Fit a GAM to each model and seed, and extract the inflection point (where the derivative is maximized)
df_emergence <- df_by_head_max_attention %>%
  group_by(mpath, seed_name) %>%
  arrange(step_modded) %>%
  mutate(log_step = log10(step_modded)) %>%
  group_map(~ {
    # Fit the GAM
    gam_model <- gam(mean_prev_self_ratio ~ s(log_step, k = 10), data = .x)

    # Get the derivative of the smooth term (the rate of change)
    gam_derivative <- predict(gam_model, type = "terms", se.fit = TRUE)

    # Find the step where the rate of change is maximized
    max_rate_of_change_step <- .x$step_modded[which.max(gam_derivative$fit)]

    # Store the results
    .x$emergence_step <- max_rate_of_change_step
    .x$predictions <- predict(gam_model, newdata = .x)
    
    return(.x)
  }) %>%
  bind_rows()

# Now plot the emergence step vs. model size
df_emerg_summ = df_emergence %>%
  group_by(model, seed, n_params) %>%
  summarise(m_step = mean(emergence_step))
```

```
## `summarise()` has grouped output by 'model', 'seed'. You can override using the
## `.groups` argument.
```

``` r
df_emerg_summ %>%
  ggplot(aes(x = n_params,
             y = m_step,
             color = model)) +
  geom_jitter() +
  scale_color_manual(values = viridisLite::viridis(4, option = "mako", 
                                                   begin = 0.8, end = 0.15)) + 
  scale_x_log10() +
  scale_y_log10() +
  theme_minimal() +
  theme(text = element_text(size = 15),
        legend.position = "bottom") +
  labs(x = "Number of Parameters (Log10)",
       y = "Max. Deriv. Step (Log10)",
       color = "") 
```

![](seed_variability_attention_anon_files/figure-html/onset-2.pdf)<!-- -->

``` r
#### Now do it with GAMs: 
### at what point does the attention~step relationship start to change?
library(mgcv)

# Fit GAM across all models/seeds
df_by_head_max_attention$log_step = log10(df_by_head_max_attention$step_modded)
gam_all <- gam(mean_prev_self_ratio ~ s(log_step), data = df_by_head_max_attention)

# Summary of the GAM
summary(gam_all)
```

```
## 
## Family: gaussian 
## Link function: identity 
## 
## Formula:
## mean_prev_self_ratio ~ s(log_step)
## 
## Parametric coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  2.19582    0.03871   56.73   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Approximate significance of smooth terms:
##               edf Ref.df     F p-value    
## s(log_step) 8.813  8.989 256.5  <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## R-sq.(adj) =    0.8   Deviance explained = 80.3%
## GCV = 0.87795  Scale est. = 0.86299   n = 576
```

``` r
# Plot the GAM curve
plot(gam_all)
```

![](seed_variability_attention_anon_files/figure-html/onset-3.pdf)<!-- -->

``` r
### Now plot it over the original models
newdat_all <- tibble(
  log_step = seq(min(df_by_head_max_attention$log_step),
                 max(df_by_head_max_attention$log_step),
                 length.out = 200)
) %>%
  mutate(step_modded = 10^log_step,
         gam_pred = predict(gam_all, newdata = .))


# Plot seed-level + average + OVERALL GAM across all facets
df_by_head_max_attention %>%
  ggplot(aes(x = step_modded,
             y = mean_prev_self_ratio,
             color = factor(seed_name))) +
  geom_line(size = .6, alpha = .4) +   # individual curves
  geom_line(data = summary_avg, 
            aes(x = step_modded, y = mean_across_seeds), 
            color = "black", size = 2) +      # per-model average
  geom_line(data = newdat_all, 
            aes(x = step_modded, y = gam_pred), 
            inherit.aes = FALSE, 
            alpha = .6,
            color = "red", size = 1.5) +        # overall GAM overlay
  theme_minimal() +
  labs(x = "Training Step (Log10)",
       y = "Max Previous/Self Attention Ratio",
       color = "") +
  scale_x_log10() +
  geom_vline(xintercept = 1000, linetype = "dotted", size = 1.2) +
  theme(text = element_text(size = 15),
        legend.position = "bottom") +
  scale_color_manual(values = viridisLite::viridis(9, option = "mako", 
                                                   begin = 0.8, end = 0.15)) + 
  facet_wrap(~reorder(model, n_params))
```

![](seed_variability_attention_anon_files/figure-html/onset-4.pdf)<!-- -->


What about reduced peak?


``` r
df_peak <- df_by_head_max_attention %>%
  group_by(mpath, model, seed_name, n_params) %>%
  summarise(max_attn = max(mean_prev_self_ratio))
```

```
## `summarise()` has grouped output by 'mpath', 'model', 'seed_name'. You can
## override using the `.groups` argument.
```

``` r
df_peak %>%
  group_by(n_params, mpath, model) %>%
  summarise(mean_max = mean(max_attn))
```

```
## `summarise()` has grouped output by 'n_params', 'mpath'. You can override using
## the `.groups` argument.
```

```
## # A tibble: 4 × 4
## # Groups:   n_params, mpath [4]
##    n_params mpath                  model       mean_max
##       <dbl> <chr>                  <chr>          <dbl>
## 1  14067712 EleutherAI/pythia-14m  pythia-14m      4.66
## 2  70426624 EleutherAI/pythia-70m  pythia-70m      4.87
## 3 162322944 EleutherAI/pythia-160m pythia-160m     6.99
## 4 405334016 EleutherAI/pythia-410m pythia-410m     8.33
```

``` r
summary(lmer(data = df_peak, 
        max_attn ~ log10(n_params) + (1|seed_name)))
```

```
## boundary (singular) fit: see help('isSingular')
```

```
## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
## lmerModLmerTest]
## Formula: max_attn ~ log10(n_params) + (1 | seed_name)
##    Data: df_peak
## 
## REML criterion at convergence: 112.5
## 
## Scaled residuals: 
##      Min       1Q   Median       3Q      Max 
## -1.78910 -0.62148 -0.07292  0.48367  2.92161 
## 
## Random effects:
##  Groups    Name        Variance  Std.Dev. 
##  seed_name (Intercept) 1.953e-21 4.419e-11
##  Residual              1.346e+00 1.160e+00
## Number of obs: 36, groups:  seed_name, 9
## 
## Fixed effects:
##                 Estimate Std. Error       df t value Pr(>|t|)    
## (Intercept)     -14.3474     2.8704  34.0000  -4.998 1.73e-05 ***
## log10(n_params)   2.5848     0.3601  34.0000   7.178 2.66e-08 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Correlation of Fixed Effects:
##             (Intr)
## lg10(n_prm) -0.998
## optimizer (nloptwrap) convergence code: 0 (OK)
## boundary (singular) fit: see help('isSingular')
```

``` r
### Plot average across d_ratio
df_peak %>%
  ggplot(aes(x = n_params,
             y = max_attn,
             color = model)) +
  geom_jitter(size = 5, alpha = .7, width = .05) +
  scale_color_manual(values = viridisLite::viridis(4, option = "mako", 
                                                   begin = 0.8, end = 0.15)) + 
  scale_x_log10() +
  theme_minimal() +
  theme(text = element_text(size = 15),
        legend.position = "bottom") +
  labs(x = "Number of Parameters (Log10)",
       y = "Max. Peak",
       color = "") 
```

![](seed_variability_attention_anon_files/figure-html/unnamed-chunk-4-1.pdf)<!-- -->

What about reduced slope?


``` r
df_by_head_max_attention <- df_by_head_max_attention %>%
  mutate(log_step = log10(step_modded))

# Fit linear models by model + seed
slopes_df <- df_by_head_max_attention %>%
  group_by(model, n_params, seed_name) %>%
  nest() %>%
  mutate(
    fit = map(data, ~ lm(mean_prev_self_ratio ~ log_step, data = .x)),
    tidied = map(fit, broom::tidy)
  ) %>%
  unnest(tidied) %>%
  filter(term == "log_step") %>%
  select(model, seed_name, estimate, std.error, statistic, p.value)
```

```
## Adding missing grouping variables: `n_params`
```

``` r
slopes_df %>%
  group_by(n_params, model) %>%
  summarise(mean_slope = mean(estimate))
```

```
## `summarise()` has grouped output by 'n_params'. You can override using the
## `.groups` argument.
```

```
## # A tibble: 4 × 3
## # Groups:   n_params [4]
##    n_params model       mean_slope
##       <dbl> <chr>            <dbl>
## 1  14067712 pythia-14m       0.742
## 2  70426624 pythia-70m       0.852
## 3 162322944 pythia-160m      1.18 
## 4 405334016 pythia-410m      1.27
```

``` r
summary(lmer(data = slopes_df, 
        estimate ~ log10(n_params) + (1|seed_name)))
```

```
## boundary (singular) fit: see help('isSingular')
```

```
## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
## lmerModLmerTest]
## Formula: estimate ~ log10(n_params) + (1 | seed_name)
##    Data: slopes_df
## 
## REML criterion at convergence: -18.2
## 
## Scaled residuals: 
##     Min      1Q  Median      3Q     Max 
## -1.9896 -0.5710 -0.1063  0.6687  2.6696 
## 
## Random effects:
##  Groups    Name        Variance Std.Dev.
##  seed_name (Intercept) 0.00000  0.0000  
##  Residual              0.02878  0.1696  
## Number of obs: 36, groups:  seed_name, 9
## 
## Fixed effects:
##                 Estimate Std. Error       df t value Pr(>|t|)    
## (Intercept)     -2.05913    0.41966 34.00000  -4.907 2.27e-05 ***
## log10(n_params)  0.38590    0.05264 34.00000   7.330 1.71e-08 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Correlation of Fixed Effects:
##             (Intr)
## lg10(n_prm) -0.998
## optimizer (nloptwrap) convergence code: 0 (OK)
## boundary (singular) fit: see help('isSingular')
```

``` r
slopes_df %>%
  ggplot(aes(x = n_params,
             y = estimate,
             color = model)) +
  geom_jitter(size = 5, alpha = .7, width = .05) +
  scale_color_manual(values = viridisLite::viridis(4, option = "mako", 
                                                   begin = 0.8, end = 0.15)) + 
  scale_x_log10() +
  theme_minimal() +
  theme(text = element_text(size = 15),
        legend.position = "bottom") +
  labs(x = "Number of Parameters (Log10)",
       y = "Slope (Attention ~ Step)",
       color = "") 
```

![](seed_variability_attention_anon_files/figure-html/unnamed-chunk-5-1.pdf)<!-- -->

## Correlation matrix



``` r
df_by_layer = df_pythia_models %>%
  group_by(model, seed_name, step_modded) %>%
  summarise(max_attn = max(mean_prev_self_ratio))
```

```
## `summarise()` has grouped output by 'model', 'seed_name'. You can override
## using the `.groups` argument.
```

``` r
df_wide <- df_by_layer %>%
  ungroup() %>%
  mutate(model_id = paste(model, "-", seed_name)) %>%
  dplyr::select(step_modded, model_id, seed_name, max_attn) %>%
  group_by(model_id, step_modded) %>%
  summarise(max_attn = mean(max_attn, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(
    names_from = model_id,
    values_from = max_attn
  )

model_id_to_params <- df_pythia_models %>%
  dplyr::select(model, seed_name, n_params) %>%
  distinct() %>%
  mutate(model_id = paste(model, "-", seed_name)) %>%
  distinct(model_id, n_params) %>%
  arrange(n_params) %>%
  pull(model_id)


### 
cor_long <- df_wide %>%
  # drop the step column; keep only model time series
  dplyr::select(-step_modded) %>%
  cor(use = "pairwise.complete.obs") %>%
  as.data.frame() %>%
  rownames_to_column("Var1") %>%
  pivot_longer(
    cols = -Var1,
    names_to = "Var2",
    values_to = "value"
  ) %>%
  mutate(
    Var1 = factor(Var1, levels = model_id_to_params),
    Var2 = factor(Var2, levels = model_id_to_params)
  )



### Make correlation matrix
cor_long %>%
  ggplot(aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  coord_fixed() +
  theme_minimal(base_size = 10) +
  scale_fill_gradient2(
    low = "blue",
    mid = "white",
    high = "red",
    midpoint = 0,
    limit = c(-1, 1),
    name = "Corr"
  ) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5),
        axis.title = element_blank()) +
  coord_fixed()
```

```
## Coordinate system already present. Adding new coordinate system, which will
## replace the existing one.
```

![](seed_variability_attention_anon_files/figure-html/mds-1.pdf)<!-- -->

``` r
### Avg. correlation within models
# extract model name without seed
cor_long <- cor_long %>%
  # extract everything before the last " - " as the model
  mutate(
    model1 = str_replace(Var1, " - .*", ""),
    model2 = str_replace(Var2, " - .*", ""),
    same_model = model1 == model2
  ) %>% 
  mutate(same_exact = Var1 == Var2) %>%
  filter(same_exact == FALSE)

cor_long = cor_long %>%
  left_join(df_pythia_models %>% select(model, n_params) %>% distinct(),
            by = c("model1" = "model")) %>%
  rename(n_params1 = n_params) %>%
  left_join(df_pythia_models %>% select(model, n_params) %>% distinct(),
            by = c("model2" = "model")) %>%
  rename(n_params2 = n_params) 


cor_long$cor = cor_long$value

summary(lm(data = cor_long,
             cor ~ same_model + log10(n_params1) + log10(n_params2)))
```

```
## 
## Call:
## lm(formula = cor ~ same_model + log10(n_params1) + log10(n_params2), 
##     data = cor_long)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -0.32125 -0.03178  0.01528  0.06006  0.10327 
## 
## Coefficients:
##                  Estimate Std. Error t value Pr(>|t|)    
## (Intercept)      0.474350   0.046498  10.202  < 2e-16 ***
## same_modelTRUE   0.078498   0.005204  15.085  < 2e-16 ***
## log10(n_params1) 0.026199   0.004070   6.437 1.73e-10 ***
## log10(n_params2) 0.026199   0.004070   6.437 1.73e-10 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.07756 on 1256 degrees of freedom
## Multiple R-squared:  0.197,	Adjusted R-squared:  0.1951 
## F-statistic: 102.7 on 3 and 1256 DF,  p-value: < 2.2e-16
```

``` r
summary(lm(data = cor_long %>% filter(same_model == FALSE),
             cor ~ log10(n_params1) + log10(n_params2)))
```

```
## 
## Call:
## lm(formula = cor ~ log10(n_params1) + log10(n_params2), data = cor_long %>% 
##     filter(same_model == FALSE))
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -0.31633 -0.03840  0.02320  0.05608  0.09835 
## 
## Coefficients:
##                   Estimate Std. Error t value Pr(>|t|)    
## (Intercept)      -0.043851   0.067229  -0.652    0.514    
## log10(n_params1)  0.058775   0.005172  11.363   <2e-16 ***
## log10(n_params2)  0.058775   0.005172  11.363   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.08165 on 969 degrees of freedom
## Multiple R-squared:  0.1666,	Adjusted R-squared:  0.1649 
## F-statistic: 96.85 on 2 and 969 DF,  p-value: < 2.2e-16
```

``` r
cor.test(cor_long$n_params1, cor_long$n_params2)
```

```
## 
## 	Pearson's product-moment correlation
## 
## data:  cor_long$n_params1 and cor_long$n_params2
## t = -1.0138, df = 1258, p-value = 0.3109
## alternative hypothesis: true correlation is not equal to 0
## 95 percent confidence interval:
##  -0.08366475  0.02669603
## sample estimates:
##         cor 
## -0.02857143
```

``` r
cor_long %>%
  group_by(same_model) %>%
  summarise(mean_cor = mean(cor),
            sd_cor = sd(cor))
```

```
## # A tibble: 2 × 3
##   same_model mean_cor sd_cor
##   <lgl>         <dbl>  <dbl>
## 1 FALSE         0.891 0.0893
## 2 TRUE          0.970 0.0317
```

``` r
cor_long %>%
  group_by(same_model, n_params1) %>%
  summarise(mean_cor = mean(cor))
```

```
## `summarise()` has grouped output by 'same_model'. You can override using the
## `.groups` argument.
```

```
## # A tibble: 8 × 3
## # Groups:   same_model [2]
##   same_model n_params1 mean_cor
##   <lgl>          <dbl>    <dbl>
## 1 FALSE       14067712    0.832
## 2 FALSE       70426624    0.930
## 3 FALSE      162322944    0.920
## 4 FALSE      405334016    0.882
## 5 TRUE        14067712    0.987
## 6 TRUE        70426624    0.974
## 7 TRUE       162322944    0.957
## 8 TRUE       405334016    0.960
```

``` r
cor_long %>%
  mutate(abs_diff_params = round(abs(log10(n_params2 - log10(n_params1))), 2)) %>%
  mutate(diff_params = abs(n_params2 - n_params1)) %>%
  filter(same_model == FALSE) %>%
  group_by(diff_params) %>%
  summarise(mean_cor = mean(cor))
```

```
## # A tibble: 6 × 2
##   diff_params mean_cor
##         <dbl>    <dbl>
## 1    56358912    0.885
## 2    91896320    0.966
## 3   148255232    0.849
## 4   243011072    0.946
## 5   334907392    0.939
## 6   391266304    0.762
```

``` r
### MDS
df_params = df_pythia_models %>%
  dplyr::select(model, n_params) %>%
  distinct() 


mds_by_step <- df_wide %>%
  drop_na() %>%
  summarise(
    mds_df = list({
      cor_mat <- cor(dplyr::select(cur_data(), starts_with("pythia-")), use = "pairwise.complete.obs")
      dist_mat <- as.dist(1 - cor_mat)
      mds <- cmdscale(dist_mat, k = 2)
      as_tibble(mds, .name_repair = "unique") %>%
        mutate(model_id = colnames(cor_mat)) 
    }),
    .groups = "drop"
  ) %>%
  unnest(mds_df) %>%
  separate(model_id, into = c("model", "seed_name", "Layer"), sep = " - ") %>%
  rename(x = `...1`, y = `...2`) %>%
  inner_join(df_params)
```

```
## New names:
## • `` -> `...1`
## • `` -> `...2`
```

```
## Warning: There was 1 warning in `summarise()`.
## ℹ In argument: `mds_df = list(...)`.
## Caused by warning:
## ! `cur_data()` was deprecated in dplyr 1.1.0.
## ℹ Please use `pick()` instead.
```

```
## Warning: Expected 3 pieces. Missing pieces filled with `NA` in 36 rows [1, 2, 3, 4, 5,
## 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, ...].
```

```
## Joining with `by = join_by(model)`
```

``` r
mds_by_step %>%
  ggplot(aes(x, y, 
           color = reorder(model, n_params))) +
  geom_jitter(size = 5, alpha = .7, width = .05) +
  theme_bw() +
  labs(x = "MDS 1",
       y = "MDS 2",
       color = "") +
  theme_minimal() +
  theme(text = element_text(size = 15),
        legend.position = "bottom") +
  scale_color_manual(values = viridisLite::viridis(4, option = "mako",
                                                   begin = 0.7, end = 0.15))
```

![](seed_variability_attention_anon_files/figure-html/mds-2.pdf)<!-- -->

``` r
# Calculate centroids for each model size
centroids <- mds_by_step %>%
  group_by(n_params) %>%
  summarise(
    centroid_x = mean(x),
    centroid_y = mean(y),
    .groups = "drop"
  )

# Calculate separation ratio for each model size
separation_stats <- centroids %>%
  rowwise() %>%
  mutate(
    # Distance to other centroids
    inter_dist = {
      current_params <- n_params
      other_centroids <- centroids %>% filter(n_params != current_params)
      mean(sqrt((other_centroids$centroid_x - centroid_x)^2 + 
                (other_centroids$centroid_y - centroid_y)^2))
    },
    # Within-cluster spread
    within_dist = {
      current_params <- n_params
      current_x <- centroid_x
      current_y <- centroid_y
      mds_by_step %>% 
        filter(n_params == current_params) %>%
        summarise(spread = mean(sqrt((x - current_x)^2 + (y - current_y)^2))) %>%
        pull(spread)
    }
  ) %>%
  mutate(sep_ratio = inter_dist / within_dist) %>%
  ungroup()

print(separation_stats)
```

```
## # A tibble: 4 × 6
##    n_params centroid_x centroid_y inter_dist within_dist sep_ratio
##       <dbl>      <dbl>      <dbl>      <dbl>       <dbl>     <dbl>
## 1  14067712   -0.144     -0.0118      0.193       0.0269      7.18
## 2  70426624    0.00743    0.00515     0.0920      0.0283      3.25
## 3 162322944    0.0288     0.0112      0.0925      0.0487      1.90
## 4 405334016    0.108     -0.00454     0.145       0.0773      1.87
```



