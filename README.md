# Project: Inventory Optimization and Demand Forecasting Using Machine Learning ⚙

## Overview

This project provides a web-based application for inventory optimization and demand forecasting, using machine learning algorithms to help businesses make informed decisions on demand forecasting and stock management.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Setup and Configuration](#setup-and-configuration)
4. [Usage](#usage)
5. [Working with the Project](#working-with-the-project)
6. [Models Used](#models-used)
7. [Contributing](#contributing)
8. [License](#license)

## Features
- **User Authentication**: Allows user registration, login, and password reset.
- **Inventory and Demand Forecasting**: Predicts future demand using various forecasting models.
- **Data Visualization**: Provides interactive charts for data analysis.
- **Dark and Light Mode Support**: Adapts to the user’s system theme for a better viewing experience.

## Installation
### Prerequisities
Ensure you have:

- **Python 3.8+**
- **pip** to install required packages.


### Step 1: Clone the Repository
```bash
git clone https://github.com/navadeep-05/INVENTORY-OPTIMIZATION-AND-DEMAND-FORECASTING-USING-MACHINE-LEARNING.git
cd INVENTORY-OPTIMIZATION-AND-DEMAND-FORECASTING-USING-MACHINE-LEARNING
```

### Step 2: Install Required Packages
All required dependencies are listed in the requirements.txt file. Run the following command to install them:
```bash
pip install -r requirements.txt
```

## Setup and Configuration
### Setting Up Database
This project uses an SQLite database for managing user accounts.
- **Database Setup**: The first time you run the application, it will automatically create a `users_data.db` file in the root directory and set up the `users` table.

## Usage
### Step 1: Start the Application
Run the app using the following command:

```bash
streamlit run ML_project_main.py
```

### Step 2: Navigate the Application
1. **Open Browser:** Once started, Streamlit will automatically open the app in a new browser tab.
2. **Landing Page:** The landing page will prompt users to log in or register if they haven't already.

## Working with the Project
### Step 1: User Registration and Login
- **Sign Up:** New users can create an account with a username, email, and password. Passwords are stored securely with hashing.
- **Login:** Existing users can log in with their credentials.
- **Password Reset:** Users can reset their password if they forget it, with the option to use their username or email.

### Step 2: Data Upload and Preprocessing
1. **Upload Data:** Users can upload their inventory or demand data in CSV format.
2. **Data Preprocessing:** The application performs basic preprocessing, including handling missing values, normalizing data, and preparing it for forecasting.

### Step 3: Model Selection and Forecasting
Users can choose from the following models:
- **Linear Regression:** Simple regression model for quick insights.
- **Random Forest Regressor:** A more complex ensemble model for capturing intricate demand patterns.
- **ARIMA Model:** Ideal for time series with seasonality.
- **LSTM(Long Short-Term Memory):** A deep learning model for capturing long-term dependencies in data
**Parameter Tuning:** Each model allows users to adjust specific parameters for improved accuracy.

### Step 4: Visualizing Forecast Results:
- **Interactive Charts:** Forecast results are displayed using plotly, providing an interactive experience where users can zoom in and explore different parts of the forecast.
- **Inventory Insights:** The visualization also offers insights on predicted demand to assist in inventory planning.

### Step 5: Interpreting Results and Making Decisions
- Users can leverage the forecasted data for strategic inventory management, such as setting reorder points, identifying seasonal trends, and optimizing stock levels.

## Models Used
- **Linear Regression:** Standard regression model for establishing a baseline.
- **Random Forest Regressor:** Ensemble learning technique for better accuracy on complex data.
- **ARIMA:** Statistical models well-suited for time series forecasting.
- **LSTM Neural Network:** Deep learning model specialized in time series forecasting with long-term dependencies.

## Contributing
Contributions are welcome! Here’s how to get started:
1. **Fork the Repository.**
2. **Create a Feature Branch:** `git checkout -b feature-name`.
3. **Commit Changes:** `git commit -m "Add feature"`.
4. **Push to Branch:** `git push origin feature-name`.
5. **Open a Pull Request.**

## License
This project is licensed under the MIT License.
