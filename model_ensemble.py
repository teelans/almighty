import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

# Load your dataset
# df = pd.read_csv('your_dataset.csv')

# Assuming your dataframe (df) is already loaded and preprocessed

# Splitting the dataset into features and target variable
X = df[['VHI_deviation', 'max_temp_deviation', 'min_temp_deviation', 'precipitation_deviation']]
y = df['yield_deviation']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "Bayesian Model": GaussianNB(),  # Gaussian Naive Bayes as a proxy for Bayesian model
    "Support Vector Regression": SVR(),
    "Gradient Boosting Machine": XGBRegressor(),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, activation='relu', solver='adam', random_state=1)
}

# Function to train and evaluate models
def train_and_evaluate(models, X_train, y_train, X_test, y_test, plot_mse=False):
    model_scores = {}
    predictions = {}

    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        # Make predictions
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        # Calculate MSE
        mse = mean_squared_error(y_test, y_pred)
        model_scores[name] = mse
        print(f"{name} MSE: {mse}")

    if plot_mse:
        plt.figure(figsize=(10, 6))
        model_names = list(model_scores.keys())
        mse_values = list(model_scores.values())
        x = np.arange(len(model_scores))  # the label locations

        plt.scatter(x, mse_values, color='darkblue', s=100)
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.xlabel('Models')
        plt.ylabel('Mean Squared Error')
        plt.title('Comparison of Model MSEs')
        plt.tight_layout()
        plt.show()

    return model_scores, predictions

# Create Ensemble Model
def create_ensemble(model_scores, model_predictions):
    total_score = sum(model_scores.values())
    weights = {name: score/total_score for name, score in model_scores.items()}
    
    ensemble_prediction = sum(weights[name] * model_predictions[name] for name in model_predictions) / sum(weights.values())
    ensemble_mse = mean_squared_error(y_test, ensemble_prediction)
    print(f"Ensemble Model MSE: {ensemble_mse}")

    return ensemble_prediction

ensemble_prediction = create_ensemble(model_scores, model_predictions)

# Example usage
model_scores, model_predictions = train_and_evaluate(models, X_train, y_train, X_test, y_test, True)

# the preference for MSE in this scenario is based on its sensitivity to large errors, which aligns well with the importance of
# accurately capturing significant deviations in crop yields due to climate anomalies.
