import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load dataset
df = pd.read_csv('data/youtube_trending_multiple_regions_with_subs.csv')

# Base features for Model 2
features = [
    'video_age_days', 'title_length_words', 'category_id', 'like_count',
    'view_count', 'like_view_ratio_percent', 'comment_count'
]

# Clean and prepare categorical variables
df['region'] = df['region'].astype(str).str.lower().fillna('unknown')
df['title_sentiment'] = df['title_sentiment'].astype(str).str.lower().fillna('neutral')

# Create all derived features for Model 2
df['engagement_rate'] = (df['like_count'] + df['comment_count']) / (df['view_count'] + 1)
df['comment_like_ratio'] = df['comment_count'] / (df['like_count'] + 1)
df['likes_per_day'] = df['like_count'] / (df['video_age_days'] + 1)
df['comments_per_day'] = df['comment_count'] / (df['video_age_days'] + 1)
df['views_per_day'] = df['view_count'] / (df['video_age_days'] + 1)
df['log_view_count'] = np.log1p(df['view_count'])
df['log_like_count'] = np.log1p(df['like_count'])
df['log_comment_count'] = np.log1p(df['comment_count'])
df['viral_score'] = np.log1p(df['views_per_day'] * df['engagement_rate'] * (df['subscriber_count'] + 1))
df['engagement_depth'] = df['comment_count'] / (df['view_count'] + 1)
df['like_engagement'] = df['like_count'] / (df['view_count'] + 1)

# Create simplified derived features to avoid encoding issues
df['category_sentiment'] = df['category_id'] * 100  # simplified
df['age_engagement'] = df['video_age_days'] * df['engagement_rate']
df['region_category_encoded'] = df['category_id'] * 100 + 50  # simplified

# Create bins
df['view_count_bins'] = pd.cut(df['view_count'], bins=5, labels=False).fillna(2)
df['age_bins'] = pd.cut(df['video_age_days'], bins=5, labels=False).fillna(2)
df['title_length_per_word'] = df['title_length_words'] / (df['title_length_words'] + 1)

# Add all derived features to feature list
derived_features = [
    'engagement_rate', 'comment_like_ratio', 'likes_per_day', 'comments_per_day',
    'views_per_day', 'log_view_count', 'log_like_count', 'log_comment_count',
    'viral_score', 'engagement_depth', 'like_engagement', 'category_sentiment',
    'age_engagement', 'region_category_encoded', 'view_count_bins', 'age_bins',
    'title_length_per_word'
]

features.extend(derived_features)

# Encode categorical variables
region_encoder = LabelEncoder()
sentiment_encoder = LabelEncoder()

df['region_encoded'] = region_encoder.fit_transform(df['region'])
df['sentiment_encoded'] = sentiment_encoder.fit_transform(df['title_sentiment'])

features.extend(['region_encoded', 'sentiment_encoded'])

# Prepare data
X = df[features].fillna(0)
y = df['subscriber_count']

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
print(f"Model 2 (Subscribers) RMSE: {rmse:,.2f}")

# Save model and scaler
with open('models/model2_subscribers.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler_model2.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Test sample prediction
sample = X.iloc[0:1]
scaled = scaler.transform(sample)
pred = model.predict(scaled)
print(f"Sample Prediction: Subscribers: {int(pred[0]):,}")