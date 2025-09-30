"""
Create sample datasets for testing and demonstration.
"""

import pandas as pd
import json
import random
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def create_sample_sentiment_data():
    """Create sample sentiment analysis data."""
    
    # Sample comments in Hindi, English, and mixed
    sample_data = [
        # Positive sentiment
        ("यह सेवा बहुत अच्छी है। डॉक्टर ने बहुत अच्छी सलाह दी।", "positive", "hi"),
        ("The online consultation was very helpful. Thank you!", "positive", "en"),
        ("Service बहुत helpful है। Doctor साहब experienced हैं।", "positive", "mixed"),
        ("मुझे यह platform बहुत पसंद आया। Very convenient!", "positive", "mixed"),
        ("Excellent service! डॉक्टर बहुत knowledgeable हैं।", "positive", "mixed"),
        ("बहुत अच्छा experience रहा। I would recommend this.", "positive", "mixed"),
        ("The video quality was good and doctor was very patient.", "positive", "en"),
        ("डॉक्टर ने properly explain किया सब कुछ। Thank you!", "positive", "mixed"),
        ("यह platform really useful है for remote consultation.", "positive", "mixed"),
        ("Great service! मुझे सब कुछ clear हो गया।", "positive", "mixed"),
        
        # Negative sentiment  
        ("यह service बहुत खराब है। डॉक्टर रूखे थे।", "negative", "hi"),
        ("I am not satisfied with the consultation. Very disappointing.", "negative", "en"),
        ("Doctor ने properly examine नहीं किया। Waste of time था।", "negative", "mixed"),
        ("बहुत bad experience रहा। Would not recommend.", "negative", "mixed"),
        ("The connection was poor and doctor seemed distracted.", "negative", "en"),
        ("Service quality बहुत poor है। Money waste हो गया।", "negative", "mixed"),
        ("डॉक्टर ने proper attention नहीं दिया। Very unprofessional.", "negative", "mixed"),
        ("यह platform user-friendly नहीं है। Too many technical issues.", "negative", "mixed"),
        ("Consultation time was too short. डॉक्टर rushed through everything.", "negative", "mixed"),
        ("बहुत disappointing था। Expected much better service quality.", "negative", "mixed"),
        
        # Neutral sentiment
        ("Service okay है। Nothing special but adequate.", "neutral", "en"),
        ("यह platform average है। कुछ खास नहीं लगा।", "neutral", "mixed"),
        ("The consultation was fine. डॉक्टर normal थे।", "neutral", "mixed"),
        ("Service normal है। Can be improved but not bad.", "neutral", "mixed"),
        ("It's an okay platform. कुछ features अच्छे हैं, कुछ नहीं।", "neutral", "mixed"),
        ("Average experience रहा। Neither good nor bad.", "neutral", "mixed"),
        ("The service is decent. डॉक्टर professional थे but nothing extraordinary.", "neutral", "mixed"),
        ("यह consultation adequate था। Met basic expectations.", "neutral", "mixed"),
        ("Standard service मिली। Could be better but acceptable.", "neutral", "mixed"),
        ("It's an average platform with basic features.", "neutral", "en"),
        
        # Additional samples for better diversity
        ("Doctor was very caring and took time to explain everything properly.", "positive", "en"),
        ("मुझे treatment plan clear नहीं लगा। Confusion में रह गया।", "negative", "mixed"),
        ("The app interface is user-friendly but needs more features.", "neutral", "en"),
        ("बहुत convenient है work from home के लिए।", "positive", "mixed"),
        ("Technical issues के कारण proper consultation नहीं हो सका।", "negative", "mixed"),
        ("Service charge reasonable है। Value for money लगा।", "positive", "mixed"),
        ("Doctor's availability limited है। Booking difficult होती है।", "negative", "mixed"),
        ("Follow-up support अच्छी है। They stayed in touch.", "positive", "mixed"),
        ("Privacy और security का proper ध्यान रखा गया है।", "positive", "mixed"),
        ("Long waiting time था। Appointment delay हो गई।", "negative", "mixed"),
    ]
    
    # Create DataFrame
    df = pd.DataFrame(sample_data, columns=['comment', 'sentiment', 'language'])
    
    # Add additional metadata
    df['id'] = range(1, len(df) + 1)
    df['length'] = df['comment'].str.len()
    df['word_count'] = df['comment'].str.split().str.len()
    
    return df


def create_sample_emotion_data():
    """Create sample emotion detection data."""
    
    sample_data = [
        # Joy
        ("यह service से मैं बहुत खुश हूं! Amazing experience!", "joy"),
        ("I'm so happy with the consultation. Doctor was wonderful!", "joy"),
        ("बहुत खुशी हुई इस platform को use करके!", "joy"),
        ("Feeling great after the consultation. Thank you so much!", "joy"),
        
        # Anger
        ("This is completely unacceptable! बहुत गुस्सा आ रहा है!", "anger"),
        ("I'm very angry with this poor service quality!", "anger"),
        ("यह क्या baat है! Such unprofessional behavior!", "anger"),
        ("Frustrated with the technical issues. Can't connect properly!", "anger"),
        
        # Fear
        ("मुझे डर लग रहा है। Is this treatment safe?", "fear"),
        ("I'm worried about the side effects of this medicine.", "fear"),
        ("Concerned about the diagnosis. Hope everything is fine.", "fear"),
        ("Anxiety हो रही है results का wait करते हुए।", "fear"),
        
        # Sadness
        ("बहुत disappointed हूं। Expected better service.", "sadness"),
        ("Feeling sad about the poor consultation experience.", "sadness"),
        ("यह service ने मुझे upset कर दिया। Very disappointing.", "sadness"),
        ("Not happy with the overall experience. Quite sad.", "sadness"),
        
        # Surprise
        ("Wow! यह तो unexpected था! Such quick response!", "surprise"),
        ("I'm amazed by the quality of service. Didn't expect this!", "surprise"),
        ("Surprising how well this platform works! Impressed!", "surprise"),
        ("यह तो shocking है! So much better than expected!", "surprise"),
        
        # Neutral
        ("The service is okay. Nothing to complain about.", "neutral"),
        ("यह platform normal है। Standard service मिली।", "neutral"),
        ("Average consultation experience. Neither good nor bad.", "neutral"),
        ("It's a decent service. Meets basic requirements.", "neutral"),
    ]
    
    # Create DataFrame
    df = pd.DataFrame(sample_data, columns=['comment', 'emotion'])
    
    # Add metadata
    df['id'] = range(1, len(df) + 1)
    df['length'] = df['comment'].str.len()
    
    return df


def create_sample_summarization_data():
    """Create sample data for summarization."""
    
    sample_data = [
        {
            "text": """
            मेरा ऑनलाइन consultation का experience बहुत अच्छा रहा। डॉक्टर ने बहुत patience से मेरी सारी problems को सुना। 
            उन्होंने detailed में मेरी condition explain की और proper medication suggest की। Video call की quality भी अच्छी थी। 
            Overall, यह service बहुत helpful है, especially उन लोगों के लिए जो hospital नहीं जा सकते। 
            I would definitely recommend this platform to others. The booking process was also very smooth.
            """,
            "summary": "ऑनलाइन consultation का अच्छा experience, डॉक्टर patient थे और proper guidance दी।"
        },
        {
            "text": """
            The online consultation platform is very user-friendly. I was able to book an appointment easily and the doctor 
            joined the call on time. The doctor was very professional and knowledgeable. They listened to all my concerns 
            carefully and provided detailed explanations. The prescription was also sent digitally which was very convenient. 
            However, the video quality could be better sometimes. Overall, I would rate this service 4 out of 5 stars. 
            I would definitely use this service again in the future.
            """,
            "summary": "User-friendly platform with professional doctors, convenient digital prescription, good overall experience."
        },
        {
            "text": """
            Initial में मुझे थोड़ी technical difficulties face करनी पड़ीं platform use करने में। But customer support team ने 
            quickly help की। Doctor consultation के time properly सब कुछ work कर रहा था। Doctor ने अच्छी advice दी लेकिन 
            follow-up के लिए physical visit recommend किया। Payment process भी smooth था। Overall decent experience रहा 
            despite initial hiccups. The platform has potential but needs some improvements in user interface.
            """,
            "summary": "Initial technical issues थे but support team helpful, doctor consultation अच्छी रही।"
        }
    ]
    
    return pd.DataFrame(sample_data)


def create_comprehensive_sample_data():
    """Create comprehensive sample data with all labels."""
    
    # Get individual datasets
    sentiment_df = create_sample_sentiment_data()
    emotion_df = create_sample_emotion_data()
    
    # Create a comprehensive dataset
    comprehensive_data = []
    
    # Combine data with multiple labels
    for i, row in sentiment_df.iterrows():
        # Assign emotions based on sentiment
        if row['sentiment'] == 'positive':
            emotion = random.choice(['joy', 'surprise'])
        elif row['sentiment'] == 'negative':
            emotion = random.choice(['anger', 'sadness', 'fear'])
        else:
            emotion = 'neutral'
            
        comprehensive_data.append({
            'id': i + 1,
            'comment': row['comment'],
            'sentiment': row['sentiment'],
            'emotion': emotion,
            'language': row['language'],
            'length': row['length'],
            'word_count': row['word_count']
        })
    
    return pd.DataFrame(comprehensive_data)


def save_sample_data():
    """Save all sample datasets to files."""
    
    # Create data directories
    data_dir = Path("data")
    sample_dir = data_dir / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating sample datasets...")
    
    # Create and save sentiment data
    sentiment_df = create_sample_sentiment_data()
    sentiment_df.to_csv(sample_dir / "sentiment_data.csv", index=False)
    sentiment_df.to_json(sample_dir / "sentiment_data.json", orient='records', indent=2, force_ascii=False)
    logger.info(f"Sentiment data saved: {len(sentiment_df)} samples")
    
    # Create and save emotion data
    emotion_df = create_sample_emotion_data()
    emotion_df.to_csv(sample_dir / "emotion_data.csv", index=False)
    emotion_df.to_json(sample_dir / "emotion_data.json", orient='records', indent=2, force_ascii=False)
    logger.info(f"Emotion data saved: {len(emotion_df)} samples")
    
    # Create and save summarization data
    summarization_df = create_sample_summarization_data()
    summarization_df.to_csv(sample_dir / "summarization_data.csv", index=False)
    summarization_df.to_json(sample_dir / "summarization_data.json", orient='records', indent=2, force_ascii=False)
    logger.info(f"Summarization data saved: {len(summarization_df)} samples")
    
    # Create and save comprehensive data
    comprehensive_df = create_comprehensive_sample_data()
    comprehensive_df.to_csv(sample_dir / "comprehensive_data.csv", index=False)
    comprehensive_df.to_json(sample_dir / "comprehensive_data.json", orient='records', indent=2, force_ascii=False)
    logger.info(f"Comprehensive data saved: {len(comprehensive_df)} samples")
    
    # Create comments.csv for backward compatibility
    sentiment_df[['comment', 'sentiment']].to_csv(sample_dir / "comments.csv", index=False)
    
    # Create a README for the sample data
    readme_content = """# Sample Data for Indian E-Consultation Analysis

This directory contains sample datasets for testing and demonstration purposes.

## Files:

1. **sentiment_data.csv/json**: Sample data for sentiment analysis
   - Contains Hindi, English, and code-mixed comments
   - Labels: positive, negative, neutral

2. **emotion_data.csv/json**: Sample data for emotion detection
   - Contains comments with emotion labels
   - Labels: joy, anger, fear, sadness, surprise, neutral

3. **summarization_data.csv/json**: Sample data for text summarization
   - Contains longer texts with reference summaries
   - Useful for evaluating summarization models

4. **comprehensive_data.csv/json**: Combined dataset with multiple labels
   - Contains sentiment, emotion, and language labels
   - Useful for multi-task evaluation

5. **comments.csv**: Simplified dataset for basic testing
   - Basic comment and sentiment pairs

## Data Statistics:

- Total sentiment samples: {sentiment_count}
- Total emotion samples: {emotion_count}
- Total summarization samples: {summarization_count}
- Languages: Hindi, English, Code-mixed
- Domain: E-consultation and healthcare feedback

## Usage:

Use these datasets to:
- Test model training and evaluation scripts
- Validate API endpoints
- Demonstrate system capabilities
- Benchmark model performance

Note: This is synthetic data created for demonstration purposes.
"""
    
    readme_path = sample_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content.format(
            sentiment_count=len(sentiment_df),
            emotion_count=len(emotion_df),
            summarization_count=len(summarization_df)
        ))
    
    logger.info("Sample data creation completed!")
    logger.info(f"Data saved in: {sample_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    save_sample_data()