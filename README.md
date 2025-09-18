# CO₂ Emissions Prediction using Linear Regression

  * This project builds a Linear Regression model to predict car CO₂ emissions based on engine size, cylinders, and fuel consumption.

# Project Workflow

1️. Import Libraries
  * pandas → Data handling
  * matplotlib & seaborn → Visualization
  * scikit-learn → Model building & evaluation

2️. Load Dataset
  * Dataset: CO2_emissions.csv
  * data = pd.read_csv("CO2_emissions.csv")

3️. Explore Data
  * Dataset info & summary
  * Check for missing values

4️. Visualize Data
  * Scatter plot → Engine Size vs CO₂ Emissions
  * Correlation heatmap → Feature relationships

5️. Feature Selection
  * Independent variables: ENGINESIZE, CYLINDERS, FUELCONSUMPTION_COMB
  * Target variable: CO2EMISSIONS

6️. Train-Test Split
  * 80% Training data
  * 20% Testing data

7️. Model Training
  * Train a Linear Regression model

8️. Predictions
  * Predict CO₂ emissions on test data

9️. Model Evaluation
  * Mean Squared Error (MSE)
  * R² Score

10. Visualization
  * Plot Actual vs Predicted emissions

 # Results
 
  * MSE → Lower is better
  * R² Score → Closer to 1.0 means stronger accuracy

# How to Run

  * Clone/download this project.
  *  Install dependencies:
     * pip install pandas matplotlib seaborn scikit-learn
  * Place CO2_emissions.csv in the project folder.
  *  Run the script:
     * python co2_regression.py

# Applications

 * Automotive research
 * Environmental analysis
 * Policy & regulations
