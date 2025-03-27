import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_and_process_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.columns = [col.strip() for col in df.columns]
        
        # Print sample of data to inspect potential issues
        print("Sample data before cleaning:")
        print(df.head(3))
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                temp_col = df[col].astype(str).str.replace(',', '')
                non_numeric_mask = ~temp_col.str.match(r'^[-+]?[0-9]*\.?[0-9]+$')
                if non_numeric_mask.any():
                    print(f"Warning: Non-numeric values found in {col} column")
                df[col] = pd.to_numeric(temp_col, errors='coerce')
                
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def add_technical_indicators(df):
    data = df.copy()
    
    ma_periods = [5, 10, 20, 50, 200]
    for period in ma_periods:
        data[f'MA_{period}'] = data['Close'].rolling(window=period).mean()
    
    ema_periods = [5, 10, 12, 20, 26, 50, 200]
    for period in ema_periods:
        data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
    
    data['Daily_Return'] = data['Close'].pct_change()
    
    for period in [5, 10, 20, 50]:
        data[f'SMA_{period}'] = data['Close'].pct_change(periods=period)*100
    
    data['Volatility_10d'] = data['Daily_Return'].rolling(window=10).std()
    data['Volatility_20d'] = data['Daily_Return'].rolling(window=20).std()
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    data['RSI'] = 100 - (100 / (1 + RS))
    
    data['BB_Middle'] = data['MA_20']
    data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
    data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']

    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    data['Next_Day_Close'] = data['Close'].shift(-1)
    data['Next_Day_Return'] = data['Next_Day_Close'] / data['Close'] - 1
    
    data_cleaned = data.dropna()
    
    print(f"Added technical indicators. Data shape: {data_cleaned.shape}")
    return data_cleaned

def prepare_model_data(data, test_size=0.2, random_state=42):
    X = data.drop(['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
                  'Next_Day_Close', 'Next_Day_Return'], axis=1)
    y_price = data['Next_Day_Close']
    y_return = data['Next_Day_Return']
    
    if X.isnull().any().any() or y_price.isnull().any() or y_return.isnull().any():
        print("Warning: Data contains null values after preparation")
    
    X_train, X_test, y_price_train, y_price_test = train_test_split(
        X, y_price, test_size=test_size, random_state=random_state)
    
    y_return_train = y_return.loc[y_price_train.index]
    y_return_test = y_return.loc[y_price_test.index]
    
    print(f"Prepared model data. Train size: {X_train.shape[0]}. Test size: {X_test.shape[0]}")
    return X_train, X_test, y_price_train, y_price_test, y_return_train, y_return_test


def build_linear_regression_model(X_train, X_test, y_train, y_test):

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    actual_direction = (y_test.pct_change() > 0).astype(int)
    predicted_direction = (pd.Series(y_pred).pct_change() > 0).astype(int)
    direction_accuracy = pd.DataFrame({
        'actual': actual_direction, 
        'predicted': predicted_direction.reset_index(drop=True)
    }).dropna()
    accuracy = (direction_accuracy['actual'] == direction_accuracy['predicted']).mean() * 100
    
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Fixed column naming for consistency
    feature_importance = pd.DataFrame({
        'feature': X_train.columns, 
        'importance': model.coef_
    })
    feature_importance['abs_importance'] = np.abs(feature_importance['importance'])
    feature_importance = feature_importance.sort_values('abs_importance', ascending=False)
    
    print(f"Linear Regression Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"R2: {r2:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Direction Accuracy: {accuracy:.2f}%")
    print(f"MAPE: {mape:.2f}%")
    print("\nTop 5 Most Important Features:")
    print(feature_importance[['feature', 'importance']].head(5))
        
    return model, y_pred, feature_importance

def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(14, 8))
    plt.plot(y_test.values, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='red')
    plt.title('Actual vs Predicted Stock Price')
    plt.xlabel('Test Data Points')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_feature_importance(feature_importance):
   
    top_features = feature_importance.head(10)
    plt.figure(figsize=(12, 8))
    plt.barh(top_features['feature'], top_features['importance'])
    plt.title('Top 10 Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def plot_performance_metrics(y_test, y_pred):
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 1, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Values')
    
    plt.subplot(2, 1, 2)
    errors = y_test.values - y_pred
    plt.hist(errors, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution')
    
    plt.tight_layout()
    plt.show()

def main():
   
    file_path = 'Netflix Inc. (NFLX) Stock Price 2002-2025.csv'
    # Load and process data
    df = load_and_process_data(file_path)
    if df is None:
        return
    
    df_with_indicators = add_technical_indicators(df)
    
    X_train, X_test, y_price_train, y_price_test, y_return_train, y_return_test = prepare_model_data(df_with_indicators)
    
    price_model, price_pred, price_feature_importance = build_linear_regression_model(
        X_train, X_test, y_price_train, y_price_test)
    
    plot_actual_vs_predicted(y_price_test, price_pred)
    plot_feature_importance(price_feature_importance)
    plot_performance_metrics(y_price_test, price_pred)
    
    print("\n--- Return Prediction Model ---")
    return_model, return_pred, return_feature_importance = build_linear_regression_model(
        X_train, X_test, y_return_train, y_return_test)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()