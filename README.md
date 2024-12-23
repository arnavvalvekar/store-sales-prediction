# Store Sales Prediction

## Overview
This project aims to predict the sales of stores based on historical data using machine learning techniques. The dataset includes various features such as store ID, promotion status, oil prices, and other factors that influence store sales. The goal is to build a model that can predict future sales with high accuracy.

## Problem Statement
The goal of this project is to predict the sales of stores given certain features using machine learning models. The evaluation metric for this competition is **Root Mean Squared Logarithmic Error (RMSLE)**.

## Dataset
The dataset consists of the following key features:
- `store_id`: Unique identifier for each store.
- `onpromotion`: Whether the product is on promotion or not.
- `cluster`: Store cluster category.
- `oil_price`: Price of oil, which can influence sales.
- `year`: Year of the sales data.
- `month`: Month of the sales data.
- `sales`: Target variable (sales of the store, which needs to be predicted).

Additional categorical features include product type, city, state, and other store-specific details.

### Data Files
- `combined_train.csv`: Training dataset.
- `test.csv`: Test dataset for predicting future sales.
  
## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/arnavvalvekar/store-sales-prediction.git
