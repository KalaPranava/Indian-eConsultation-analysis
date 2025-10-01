#!/usr/bin/env python3
"""
Test script for the lightweight ML API
"""
from scripts.lightweight_ml_api import analyze_sentiment_lightweight, analyze_emotion_lightweight, summarize_text_lightweight
import time

print('🧪 Testing Complete Lightweight API')
print('=' * 50)

# Test sentiment
print('📊 Testing Sentiment Analysis...')
texts = [
    'This service is excellent and very helpful!',
    'बहुत अच्छी सेवा है, धन्यवाद',
    'I am very disappointed with the treatment'
]

for text in texts:
    result = analyze_sentiment_lightweight(text)
    print(f'  Text: {text[:30]}...')
    print(f'  Sentiment: {result["sentiment"]} (confidence: {result["confidence"]})')
    print(f'  Method: {result["method"]}')
    print()

# Test emotion
print('😊 Testing Emotion Analysis...')
emotion_result = analyze_emotion_lightweight('I am so happy with the doctor')
print(f'  Primary emotion: {emotion_result["primary_emotion"]}')
print(f'  Confidence: {emotion_result["confidence"]}')
print()

# Test summarization
print('📝 Testing Summarization...')
long_text = 'The doctor was very professional and caring. The consultation was thorough and helpful. I received good treatment and medication. The staff was also very supportive. Overall, I am satisfied with the service and would recommend it to others.'
summary_result = summarize_text_lightweight(long_text, 50)
print(f'  Original: {len(long_text)} chars')
print(f'  Summary: {len(summary_result["summary"])} chars')
print(f'  Compression: {summary_result["compression_ratio"]}')
print()

print('✅ All tests completed successfully!')
print(f'💾 Estimated total memory usage: ~50MB')