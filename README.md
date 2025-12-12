What’s Cooking – Kaggle Project

Overview This repository contains my solution for the Kaggle competition What’s Cooking?. The challenge is to predict the type of cuisine (e.g., Italian, Indian, Mexican) given a list of ingredients. It’s a multi‑class classification problem that highlights the importance of feature engineering and text‑based modeling in machine learning.

Dataset

Source: https://www.kaggle.com/c/whats-cooking

Description:

train.json – Training dataset with recipes and labeled cuisines.

test.json – Test dataset for predictions.

sampleSubmission.csv – Example submission format.

Features: Each recipe is represented by a list of ingredients.

Target: Cuisine label for each recipe.

Methodology Implemented in R:

Data preprocessing: parsing JSON files, cleaning ingredient lists, encoding categorical features.

Modeling approaches: multinomial logistic regression, random forests, and ensemble methods.

Evaluation: Kaggle leaderboard metric is accuracy. Predictions saved in CSV format for submission.

Reproducibility: scripts included for reruns and extensions.

Results

Predictions generated and saved in submission files.

Models achieved competitive scores on Kaggle’s leaderboard.

Feature engineering on ingredient frequency and presence improved classification accuracy.

Repository structure ├── WhatsCooking.R # Main R script with analysis and modeling ├── train.json # Training dataset ├── test.json # Test dataset ├── sampleSubmission.csv # Kaggle sample submission format └── README.md # Project documentation

How to run

Clone the repository: git clone https://github.com/supermaxaroni/WhatsCooking.git

Open WhatsCooking.R in RStudio or run via R console.

Install required packages: install.packages(c("jsonlite", "randomForest", "caret", "data.table"))

Execute the script to generate predictions.

Submission files (*.csv) can be uploaded directly to Kaggle.

Future work

Explore gradient boosting methods (XGBoost, LightGBM).

Implement stacking/ensembles for improved accuracy.

Perform deeper feature engineering on ingredient combinations.

Automate hyperparameter tuning for Random Forests.

Acknowledgments

Kaggle for hosting the competition.

R community packages (jsonlite, randomForest, caret, data.table) that made modeling efficient.

Inspiration from Kaggle kernels and discussions.
