import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# Create models directory if it doesn't exist
os.makedirs('../models', exist_ok=True)

# Load dataset
df = pd.read_csv('data/youtube_trending_multiple_regions_with_subs.csv')

# Clean and prepare categorical variables
df['region'] = df['region'].astype(str).str.lower().fillna('unknown')
df['title_sentiment'] = df['title_sentiment'].astype(str).str.lower().fillna('neutral')

# Base features for Model 4
features = [
    'video_age_days', 'category_id', 'like_count', 'view_count',
    'comment_count', 'subscriber_count'
]

# Create derived features for Model 4
df['engagement_rate'] = (df['like_count'] + df['comment_count']) / (df['view_count'] + 1)
df['views_per_day'] = df['view_count'] / (df['video_age_days'] + 1)
df['viral_score'] = np.log1p(df['views_per_day'] * df['engagement_rate'] * (df['subscriber_count'] + 1))
df['conversion_rate'] = df['view_count'] / (df['subscriber_count'] + 1)
df['likes_per_day'] = df['like_count'] / (df['video_age_days'] + 1)
df['sub_growth_rate'] = (df['subscriber_count'] + 1) / (df['video_age_days'] + 1)

# Add derived features to feature list
derived_features = [
    'engagement_rate', 'views_per_day', 'viral_score', 'conversion_rate',
    'likes_per_day', 'sub_growth_rate'
]
features.extend(derived_features)

# Create trending status based on views_per_day and engagement_rate
trending_threshold = df['views_per_day'].quantile(0.7)
engagement_threshold = df['engagement_rate'].quantile(0.6)

df['trending_status'] = np.where(
    (df['views_per_day'] >= trending_threshold) & (df['engagement_rate'] >= engagement_threshold),
    'successful', 'unsuccessful'
)

# Encode categorical variables
region_encoder = LabelEncoder()
sentiment_encoder = LabelEncoder()

df['region_encoded'] = region_encoder.fit_transform(df['region'])
df['sentiment_encoded'] = sentiment_encoder.fit_transform(df['title_sentiment'])

features.extend(['region_encoded', 'sentiment_encoded'])

# Prepare data
X = df[features].fillna(0)
y = df['trending_status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model 4 (Trending Status) Accuracy: {accuracy:.4f}")

# Save model
with open('models/model4_trending_days.pkl', 'wb') as f:
    pickle.dump(model, f)

# Test sample prediction
sample = X.iloc[0:1]
pred = model.predict(sample)
print(f"Sample Prediction: Status: {pred[0]}")