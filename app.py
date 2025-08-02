from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load all models and encoders
models = {}
scalers = {}
encoders = {}

def load_models():
    # Load models
    with open('models/model1_view_count.pkl', 'rb') as f:
        models['model1'] = pickle.load(f)
    
    with open('models/model2_subscribers.pkl', 'rb') as f:
        models['model2'] = pickle.load(f)
    
    with open('models/model3_popularity_class.pkl', 'rb') as f:
        models['model3'] = pickle.load(f)
    
    with open('models/model4_trending_days.pkl', 'rb') as f:
        models['model4'] = pickle.load(f)
    
    # Load scalers
    with open('models/scaler_model1.pkl', 'rb') as f:
        scalers['model1'] = pickle.load(f)
    
    with open('models/scaler_model2.pkl', 'rb') as f:
        scalers['model2'] = pickle.load(f)
    
    # Load encoders
    with open('models/region_encoder.pkl', 'rb') as f:
        encoders['region'] = pickle.load(f)
    
    with open('models/sentiment_encoder.pkl', 'rb') as f:
        encoders['sentiment'] = pickle.load(f)

# Load models on startup
load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_name = request.args.get('model')
    
    if not model_name or model_name not in models:
        return jsonify({'error': 'Invalid model specified'}), 400
    
    try:
        # Clean and encode categorical variables (same as training)
        region_clean = str(data['region']).lower() if data['region'] else 'unknown'
        sentiment_clean = str(data['title_sentiment']).lower() if data['title_sentiment'] else 'neutral'
        
        # Handle unknown categories by using the first category if not found
        try:
            region_encoded = encoders['region'].transform([region_clean])[0]
        except ValueError:
            region_encoded = 0  # fallback for unknown regions
            
        try:
            sentiment_encoded = encoders['sentiment'].transform([sentiment_clean])[0]
        except ValueError:
            sentiment_encoded = 0  # fallback for unknown sentiments
        
        if model_name == 'model1':
            # Model 1: View Count Predictor
            # Derived features
            likes_per_day = data['like_count'] / (data['video_age_days'] + 1)
            like_view_ratio = data['like_count'] / (data['view_count'] + 1)
            
            features = np.array([[
                data['video_age_days'], data['title_length_words'], data['category_id'],
                data['like_count'], data['like_view_ratio_percent'], data['subscriber_count'],
                likes_per_day, like_view_ratio, region_encoded, sentiment_encoded
            ]])
            
            features_scaled = scalers['model1'].transform(features)
            prediction = models['model1'].predict(features_scaled)[0]
            result = f"Views: {int(prediction):,}"
            
        elif model_name == 'model2':
            # Model 2: Subscribers Predictor
            # All derived features for Model 2
            engagement_rate = (data['like_count'] + data['comment_count']) / (data['view_count'] + 1)
            comment_like_ratio = data['comment_count'] / (data['like_count'] + 1)
            likes_per_day = data['like_count'] / (data['video_age_days'] + 1)
            comments_per_day = data['comment_count'] / (data['video_age_days'] + 1)
            views_per_day = data['view_count'] / (data['video_age_days'] + 1)
            log_view_count = np.log1p(data['view_count'])
            log_like_count = np.log1p(data['like_count'])
            log_comment_count = np.log1p(data['comment_count'])
            viral_score = np.log1p(views_per_day * engagement_rate * (data['subscriber_count'] + 1))
            engagement_depth = data['comment_count'] / (data['view_count'] + 1)
            like_engagement = data['like_count'] / (data['view_count'] + 1)
            category_sentiment = data['category_id'] * 100  # simplified
            age_engagement = data['video_age_days'] * engagement_rate
            region_category_encoded = data['category_id'] * 100 + 50  # simplified
            view_count_bins = 2  # middle bin as default
            age_bins = 2  # middle bin as default
            title_length_per_word = data['title_length_words'] / (data['title_length_words'] + 1)
            
            features = np.array([[
                data['video_age_days'], data['title_length_words'], data['category_id'],
                data['like_count'], data['view_count'], data['like_view_ratio_percent'],
                data['comment_count'], engagement_rate, comment_like_ratio, likes_per_day,
                comments_per_day, views_per_day, log_view_count, log_like_count,
                log_comment_count, viral_score, engagement_depth, like_engagement,
                category_sentiment, age_engagement, region_category_encoded,
                view_count_bins, age_bins, title_length_per_word, region_encoded, sentiment_encoded
            ]])
            
            features_scaled = scalers['model2'].transform(features)
            prediction = models['model2'].predict(features_scaled)[0]
            result = f"Subscribers: {int(prediction):,}"
            
        elif model_name == 'model3':
            # Model 3: Popularity Class
            features = np.array([[
                data['video_age_days'], data['category_id'], data['like_count'],
                data['view_count'], data['comment_count'], data['subscriber_count'],
                region_encoded, sentiment_encoded
            ]])
            
            prediction = models['model3'].predict(features)[0]
            result = f"Class: {prediction}"
            
        elif model_name == 'model4':
            # Model 4: Trending Status
            # Derived features for Model 4
            engagement_rate = (data['like_count'] + data['comment_count']) / (data['view_count'] + 1)
            views_per_day = data['view_count'] / (data['video_age_days'] + 1)
            viral_score = np.log1p(views_per_day * engagement_rate * (data['subscriber_count'] + 1))
            conversion_rate = data['view_count'] / (data['subscriber_count'] + 1)
            likes_per_day = data['like_count'] / (data['video_age_days'] + 1)
            sub_growth_rate = (data['subscriber_count'] + 1) / (data['video_age_days'] + 1)
            
            features = np.array([[
                data['video_age_days'], data['category_id'], data['like_count'],
                data['view_count'], data['comment_count'], data['subscriber_count'],
                engagement_rate, views_per_day, viral_score, conversion_rate,
                likes_per_day, sub_growth_rate, region_encoded, sentiment_encoded
            ]])
            
            prediction = models['model4'].predict(features)[0]
            result = f"Status: {prediction}"
        
        return jsonify({'prediction': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)