library(tidyverse)
library(tidymodels)
library(textrecipes)
library(jsonlite)
library(stacks)
library(xgboost)
library(discrim)
library(doParallel)

# ===============================================
# Helper functions
# ===============================================
clean_text <- function(x) {
  x %>%
    tolower() %>%
    str_replace_all("[0-9]", " ") %>%
    str_replace_all("[[:punct:]]", " ") %>%
    str_replace_all("\\s+", " ") %>%
    str_trim()
}

clean_ingredient_vector <- function(ing_vec) {
  ing_vec %>%
    map_chr(clean_text)
}

# ===============================================
# Load data
# ===============================================
train_raw <- read_json("train.json", simplifyVector = TRUE)
test_raw  <- read_json("test.json", simplifyVector = TRUE)

# ===============================================
# Clean ingredients text
# ===============================================
train_raw <- train_raw %>%
  mutate(
    ingredients_text = map_chr(
      ingredients, ~ paste(clean_ingredient_vector(.x), collapse = " ")
    )
  )

test_raw <- test_raw %>%
  mutate(
    ingredients_text = map_chr(
      ingredients, ~ paste(clean_ingredient_vector(.x), collapse = " ")
    )
  )

write_csv(test_raw, "test_new.csv")
write_csv(train_raw, "train_new.csv")

# ===============================================
# Split training (initial_split)
# ===============================================
set.seed(123)
split <- initial_split(train_raw, prop = 0.9, strata = cuisine)

train <- training(split)
valid <- testing(split)

# ===============================================
# Text recipe (smaller token limit)
# ===============================================
rec <- recipe(cuisine ~ ingredients_text, data = train) %>%
  step_tokenize(ingredients_text) %>%
  step_tokenfilter(ingredients_text, max_tokens = 2000) %>%
  step_tfidf(ingredients_text)

# ===============================================
# Models for ensemble
# ===============================================
glmnet_spec <- multinom_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet")

nb_spec <- naive_Bayes(smoothness = tune()) %>%
  set_engine("naivebayes")

xgb_spec <- boost_tree(
  trees = tune(),
  learn_rate = tune(),
  mtry = tune(),
  tree_depth = tune()
) %>% set_engine("xgboost") %>% set_mode("classification")

# ===============================================
# Workflows
# ===============================================
wf_glmnet <- workflow() %>% add_recipe(rec) %>% add_model(glmnet_spec)
wf_nb     <- workflow() %>% add_recipe(rec) %>% add_model(nb_spec)
wf_xgb    <- workflow() %>% add_recipe(rec) %>% add_model(xgb_spec)

# ===============================================
# Parallel backend
# ===============================================
cl <- makeCluster(parallel::detectCores() - 1)
registerDoParallel(cl)

# ===============================================
# Tune with small grids
# ===============================================
glmnet_res <- tune_grid(
  wf_glmnet,
  resamples = vfold_cv(train, v = 2, strata = cuisine),
  grid = 1
)

nb_res <- tune_grid(
  wf_nb,
  resamples = vfold_cv(train, v = 2, strata = cuisine),
  grid = 1
)

xgb_res <- tune_grid(
  wf_xgb,
  resamples = vfold_cv(train, v = 2, strata = cuisine),
  grid = 2
)

# ===============================================
# Stacking ensemble
# ===============================================
model_stack <- stacks() %>%
  add_candidates(glmnet_res) %>%
  add_candidates(nb_res) %>%
  add_candidates(xgb_res) %>%
  blend_predictions() %>%
  fit_members()

# ===============================================
# Final fit on all training data
# ===============================================
final_fit <- fit(model_stack, data = train_raw)

# ===============================================
# Predict on test
# ===============================================
test_preds <- predict(final_fit, test_raw)

submission <- tibble(
  id = test_raw$id,
  cuisine = test_preds$.pred_class
)

write_csv(submission, "submission.csv")
cat("submission.csv created!\n")

# Stop cluster
stopCluster(cl)
