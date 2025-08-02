import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import os

# Create models directory if it doesn't exist
os.makedirs('/models', exist_ok=True)

# Load dataset
df = pd.read_csv('data/youtube_trending_multiple_regions_with_subs.csv')

# Prepare features for Model 1
features = [
    'video_age_days', 'title_length_words', 'category_id', 'like_count',
    'like_view_ratio_percent', 'subscriber_count'
]

# Create derived features
df['likes_per_day'] = df['like_count'] / (df['video_age_days'] + 1)
df['like_view_ratio'] = df['like_count'] / (df['view_count'] + 1)

# Add derived features to feature list
features.extend(['likes_per_day', 'like_view_ratio'])

# Clean and prepare categorical variables
df['region'] = df['region'].astype(str).str.lower().fillna('unknown')
df['title_sentiment'] = df['title_sentiment'].astype(str).str.lower().fillna('neutral')

# Encode categorical variables
region_encoder = LabelEncoder()
sentiment_encoder = LabelEncoder()

df['region_encoded'] = region_encoder.fit_transform(df['region'])
df['sentiment_encoded'] = sentiment_encoder.fit_transform(df['title_sentiment'])

features.extend(['region_encoded', 'sentiment_encoded'])

# Prepare data
X = df[features].fillna(0)
y = df['view_count']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model 1 (View Count) RMSE: {rmse:,.2f}")

# Save model and scaler
with open('models/model1_view_count.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler_model1.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/region_encoder.pkl', 'wb') as f:
    pickle.dump(region_encoder, f)

with open('models/sentiment_encoder.pkl', 'wb') as f:
    pickle.dump(sentiment_encoder, f)

# Test sample prediction
sample = X.iloc[0:1]
scaled = scaler.transform(sample)
pred = model.predict(scaled)
print(f"Sample Prediction: Views: {int(pred[0]):,}")