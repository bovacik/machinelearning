# This cell loads the procurement_data.csv file and displays the first few rows to inspect its structure
import pandas as pd

df_procurement = pd.read_csv('/Users/bovacik/Downloads/Github/procurement_data.csv')

# Print first few rows to inspect structure
print(df_procurement.head())

# This cell applies Lasso and Ridge regression on procurement_data.csv.
# We assume that the target variable is 'Final Purchase Price' and the remaining numeric features are predictors.

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the data
# df_procurement already loaded from previous cell; if not, load again

# Define the target and feature columns
# We ignore categorical variables like 'Item Category' for now

features = ['Quantity', 'Supplier Experience (Years)', 'Urgency (Days to Delivery)', 'Negotiation Effort (Hours)', 'Previous Price Paid', 'Discount Applied (%)']
target = 'Final Purchase Price'

# Check if these columns exist in our dataframe
missing_cols = [col for col in features + [target] if col not in df_procurement.columns]
if missing_cols:
    print('Missing columns: ' + str(missing_cols))
else:
    # Split data
    X = df_procurement[features]
    y = df_procurement[target]

    # Create a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    lasso = Lasso(alpha=0.1, random_state=42)
    ridge = Ridge(alpha=1.0, random_state=42)

    # Fit models on training data
    lasso.fit(X_train, y_train)
    ridge.fit(X_train, y_train)

    # Predictions
    y_pred_lasso = lasso.predict(X_test)
    y_pred_ridge = ridge.predict(X_test)

    # Calculate Mean Squared Error
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)

    print('Lasso Regression Coefficients:')
    for coef, name in zip(lasso.coef_, features):
        print(name + ': ' + str(coef))
    print('Lasso Regression intercept: ' + str(lasso.intercept_))
    print('Lasso Mean Squared Error: ' + str(mse_lasso))

    print('\
Ridge Regression Coefficients:')
    for coef, name in zip(ridge.coef_, features):
        print(name + ': ' + str(coef))
    print('Ridge Regression intercept: ' + str(ridge.intercept_))
    print('Ridge Mean Squared Error: ' + str(mse_ridge))

# This cell visualizes the model predictions for both Lasso and Ridge regression
import matplotlib.pyplot as plt

# We already computed predictions: y_pred_lasso and y_pred_ridge, and split data X_test and y_test

plt.figure(figsize=(12, 5))

# Lasso predictions
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lasso, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Final Purchase Price')
plt.ylabel('Predicted Final Purchase Price')
plt.title('Lasso Regression Predictions')

# Ridge predictions
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_ridge, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Final Purchase Price')
plt.ylabel('Predicted Final Purchase Price')
plt.title('Ridge Regression Predictions')

plt.tight_layout()
plt.show()

print('Displayed scatter plots for Lasso and Ridge predictions with a reference line.')

# Visualize feature importance for both models
import numpy as np
import matplotlib.pyplot as plt

features = ['Quantity', 'Supplier Experience (Years)', 'Urgency (Days to Delivery)', 
           'Negotiation Effort (Hours)', 'Previous Price Paid', 'Discount Applied (%)']

# Get absolute coefficients for comparison
lasso_importance = np.abs(lasso.coef_)
ridge_importance = np.abs(ridge.coef_)

# Create bar plot
plt.figure(figsize=(12, 6))
x = np.arange(len(features))
width = 0.35

plt.bar(x - width/2, lasso_importance, width, label='Lasso', alpha=0.8)
plt.bar(x + width/2, ridge_importance, width, label='Ridge', alpha=0.8)

plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Value')
plt.title('Feature Importance Comparison: Lasso vs Ridge')
plt.xticks(x, features, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# Analyze non-linear relationships using scatter plots and polynomial features
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Create a figure with multiple subplots
plt.figure(figsize=(15, 10))

# Key numeric features
features = ['Quantity', 'Supplier Experience (Years)', 'Previous Price Paid', 'Discount Applied (%)']
target = 'Final Purchase Price'

for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    
    # Scatter plot with regression line
    sns.regplot(data=df_procurement, x=feature, y=target, 
                scatter_kws={'alpha':0.5}, 
                order=2,  # Fit a polynomial of degree 2
                line_kws={'color': 'red'})
    
    plt.title(f'{feature} vs {target}')

plt.tight_layout()
plt.show()

# Calculate Spearman correlation (captures monotonic relationships, including non-linear)
correlations = []
for feature in features:
    correlation, _ = stats.spearmanr(df_procurement[feature], df_procurement[target])
    correlations.append((feature, correlation))

print("\
Spearman Correlations with Final Purchase Price:")
for feature, corr in correlations:
    print(f"{feature}: {corr:.3f}")

# Analyze the impact of discounts in detail
import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot of Discount Applied (%) vs Final Purchase Price
plt.figure(figsize=(10, 6))
plt.scatter(df_procurement['Discount Applied (%)'], df_procurement['Final Purchase Price'], alpha=0.6, color='purple')
plt.xlabel('Discount Applied (%)')
plt.ylabel('Final Purchase Price')
plt.title('Discounts vs Final Purchase Price')
plt.show()

# Box plot to understand distribution effects (group by discount buckets)
bins = [-float('inf'), 0, 10, 20, 30, float('inf')]  # adjust bins if needed
df_procurement['Discount_Bin'] = pd.cut(df_procurement['Discount Applied (%)'], bins=bins)

plt.figure(figsize=(10, 6))
sns.boxplot(x='Discount_Bin', y='Final Purchase Price', data=df_procurement, palette='coolwarm')
plt.xlabel('Discount Applied (%) Bins')
plt.ylabel('Final Purchase Price')
plt.title('Impact of Discount Bins on Final Purchase Price')
plt.xticks(rotation=45)
plt.show()

# Print correlation specifically for discounts using Pearson and Spearman correlations
discount_pearson = df_procurement[['Discount Applied (%)', 'Final Purchase Price']].corr().iloc[0,1]
from scipy.stats import spearmanr
discount_spearman, _ = spearmanr(df_procurement['Discount Applied (%)'], df_procurement['Final Purchase Price'])

print('Pearson correlation between Discount Applied (%) and Final Purchase Price: ' + str(discount_pearson))
print('Spearman correlation between Discount Applied (%) and Final Purchase Price: ' + str(discount_spearman))

# Summary analysis:
# Analyzing the distribution of discounts compared to final purchase price provides insight into how discounts affect pricing.

# The scatter plot shows a negative trend, indicating that as the discount increases, the final purchase price tends to drop. 

# The box plot reveals the spread of purchase prices within discount bins, which may highlight outliers or variability within each discount category.

# The Pearson and Spearman correlations quantify the linear and monotonic relationships, respectively, between discounts and prices.

# Analyze the relationship between Quantity and Discount Applied (%)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Scatter plot: Quantity vs Discount Applied
plt.figure(figsize=(10, 6))
plt.scatter(df_procurement['Quantity'], df_procurement['Discount Applied (%)'], alpha=0.6, color='teal')
plt.xlabel('Quantity')
plt.ylabel('Discount Applied (%)')
plt.title('Scatter Plot: Quantity vs Discount Applied (%)')
plt.show()

# To further study, create bins for Quantity and plot boxplots of Discount per bin
bins = np.linspace(df_procurement['Quantity'].min(), df_procurement['Quantity'].max(), 6)  
df_procurement['Quantity_Bin'] = pd.cut(df_procurement['Quantity'], bins=bins, include_lowest=True)

plt.figure(figsize=(10, 6))
sns.boxplot(x='Quantity_Bin', y='Discount Applied (%)', data=df_procurement, palette='viridis')
plt.xlabel('Quantity Bins')
plt.ylabel('Discount Applied (%)')
plt.title('Box Plot of Discount by Quantity Bins')
plt.xticks(rotation=45)
plt.show()

# Calculate Pearson and Spearman correlations
from scipy.stats import pearsonr, spearmanr

pearson_corr, p_value_pearson = pearsonr(df_procurement['Quantity'], df_procurement['Discount Applied (%)'])
spearman_corr, p_value_spearman = spearmanr(df_procurement['Quantity'], df_procurement['Discount Applied (%)'])

print('Pearson correlation between Quantity and Discount Applied (%): ' + str(pearson_corr))
print('Spearman correlation between Quantity and Discount Applied (%): ' + str(spearman_corr))