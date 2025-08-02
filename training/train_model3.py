import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# Create models directory if it doesn't exist
os.makedirs('/models', exist_ok=True)

# Load dataset
df = pd.read_csv('data/youtube_trending_multiple_regions_with_subs.csv')

# Features for Model 3
features = [
    'video_age_days', 'category_id', 'like_count', 'view_count',
    'comment_count', 'subscriber_count'
]

# Clean and prepare categorical variables
df['region'] = df['region'].astype(str).str.lower().fillna('unknown')
df['title_sentiment'] = df['title_sentiment'].astype(str).str.lower().fillna('neutral')

# Create popularity classes based on view count percentiles
view_percentiles = df['view_count'].quantile([0.33, 0.66])
df['popularity_class'] = pd.cut(df['view_count'], 
                               bins=[-np.inf, view_percentiles[0.33], view_percentiles[0.66], np.inf],
                               labels=['low', 'medium', 'high'])

# Encode categorical variables
region_encoder = LabelEncoder()
sentiment_encoder = LabelEncoder()

df['region_encoded'] = region_encoder.fit_transform(df['region'])
df['sentiment_encoded'] = sentiment_encoder.fit_transform(df['title_sentiment'])

features.extend(['region_encoded', 'sentiment_encoded'])

# Prepare data
X = df[features].fillna(0)
y = df['popularity_class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model 3 (Popularity Class) Accuracy: {accuracy:.4f}")

# Save model
with open('models/model3_popularity_class.pkl', 'wb') as f:
    pickle.dump(model, f)

# Test sample prediction
sample = X.iloc[0:1]
pred = model.predict(sample)
print(f"Sample Prediction: Class: {pred[0]}")