# INVENTORY OPTIMIZATION AND DEMAND FORECASTING USING MACHINE LEARNING ⚙
## Overview
This project focuses on using machine learning techniques to enhance inventory management and accurately demand of sales. It's goal is to support businesses, such as retail stores, shopkeepers and supply chains, in maintaining optimal inventory levels , minimizing stockouts, and reducing excess inventory. 

## Table of Contents
1. Features
2. Installation
3. Setup and configuration
4. Usage
5. Working with the Project
6. Models used
7. Contributing
8. License

## Features
- **User Authentication**: Allows user registration, login, and password reset.
- **Inventory and Demand Forecasting**: Predicts future demand using various forecasting models.
- **Data Visualization**: Provides interactive charts for data analysis.
- **Dark and Light Mode Support**: Adapts to the user’s system theme for a better viewing experience.

## Installation
### Prerequisities
- Python 3.5+
- pip to install required packages
### Step 1: Clone the Repository
\bash git clone https://github.com/navadeep-05/INVENTORY-OPTIMIZATION-AND-DEMAND-FORECASTING-USING-MACHINE-LEARNING.git
cd INVENTORY-OPTIMIZATION-AND-DEMAND-FORECASTING-USING-MACHINE-LEARNING
\
### Step 2: Install Required Packages
All required dependencies are listed in the requirements.txt file. Run the following command to install them:
\bash pip install -r requirements.txt\

## Setup and Configuration
### Setting Up Database
This project uses an SQLite database for managing user accounts.
- **Database Setup**: The first time you run the application, it will automatically create a `users_data.db` file in the root directory and set up the `users` table.

## Usage
### Step 1: Start the Application
Run the app using the following command:

```bash
streamlit run ML_project_main.py
