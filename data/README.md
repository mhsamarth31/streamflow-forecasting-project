# Streamflow Forecasting Using Artificial Neural Networks

## Project Overview

This project implements and evaluates two neural-network-based approaches for daily streamflow forecasting:

1. Artificial Neural Network (ANN)
2. Long Short-Term Memory (LSTM)

The implementation is based on the methodology described in the research paper:

**Application of Artificial Neural Network in Streamflow Forecasting**  
**Authors:** Mian Xiao and Shanni You

The objective of this project is to use historical rainfall and streamflow observations to forecast future streamflow. The project implements the core forecasting methodology described in the paper and compares the performance of a standard feed-forward ANN with an LSTM model.

---

## Problem Statement

Streamflow forecasting is an important problem in hydrology and water-resource management. Accurate prediction of future streamflow can support applications such as flood management, reservoir operation, and water-resource planning.

Given historical rainfall and streamflow observations, the objective is to learn a relationship between previous observations and the current streamflow:

Q(t) = f(R(t-1), ..., R(t-m), Q(t-1), ..., Q(t-n))

where:

- Q(t) = streamflow at time t
- R(t) = rainfall at time t
- m = rainfall lag
- n = streamflow lag

This project investigates the use of a standard feed-forward ANN and an LSTM model to learn this relationship and forecast streamflow.

---

## Objectives

The main objectives of this project are:

- To preprocess historical rainfall and streamflow data.
- To construct supervised learning samples using historical lagged observations.
- To implement a feed-forward Artificial Neural Network.
- To implement an LSTM-based sequence model.
- To tune important hyperparameters of both models.
- To train the models on historical streamflow data.
- To evaluate the models using Root Mean Squared Error (RMSE).
- To compare ANN and LSTM forecasting performance.
- To visualize actual versus predicted streamflow.
- To analyze model training behaviour and forecasting performance.

---

## Reference Paper

**Title:** Application of Artificial Neural Network in Streamflow Forecasting

**Authors:** Mian Xiao and Shanni You

The reference paper investigates streamflow forecasting using a standard ANN and an LSTM model. The methodology implemented in this project follows the main modeling approach described in the paper.

The paper investigates the effect of factors including:

- Learning rate
- Number of hidden neurons
- Rainfall lag
- Runoff lag
- LSTM sequence length

The paper evaluates the models using RMSE and also investigates peak and non-peak streamflow prediction.

---

# Dataset

## Dataset Description

The dataset used in this project contains daily observations with the following columns:

| Column | Description |
|---|---|
| `Date` | Observation date |
| `Rainfall` | Daily rainfall observation |
| `Streamflow` | Daily streamflow observation |

The complete dataset contains approximately **8,226 daily observations**.

### Complete Dataset Period

- Start Date: **June 1, 1993**
- End Date: **October 8, 2018**

The dataset contains no missing values and no duplicate dates.

---

## Dataset Used for the Main Experiment

To align the experimental period with the reference paper, the main experiment uses the period:

**2003–2012**

This period contains **3,653 daily observations**.

The data is divided chronologically into:

### Training Period

**2003–2007**

### Validation Period

**2008–2012**

No random shuffling is performed because streamflow forecasting is a time-series problem.

The chronological split ensures that the models are trained using earlier observations and evaluated on later observations.

---

# Data Preprocessing

The following preprocessing steps are performed:

1. The dataset is sorted chronologically.
2. Rainfall and streamflow values are extracted.
3. Historical lagged observations are created.
4. Training and validation samples are generated based on the target date.
5. Input values are normalized before model training.
6. Scaling parameters are obtained from the training period.
7. The same scaling parameters are applied to the validation period.
8. Predictions are converted back to the original streamflow scale for evaluation.

The implementation constructs the lag/sequence samples from the chronological data before separating the target dates into training and validation periods. This allows early validation samples to use valid historical observations from the end of the training period.

---

# Methodology

## 1. Artificial Neural Network

The ANN is implemented as a feed-forward fully connected neural network.

The architecture consists of:

```text
Input Layer
     |
     v
Hidden Layer
     |
     v
Output Layer