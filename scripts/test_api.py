"""
Test script for the working API
"""
import requests
import json

def test_api():
    base_url = "http://127.0.0.1:8000"
    
    print("🧪 Testing Indian E-Consultation Analysis API")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health Check: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")
        return
    
    print()
    
    # Test sentiment analysis
    try:
        text = "यह सेवा बहुत अच्छी है। The doctor gave excellent advice!"
        response = requests.post(f"{base_url}/analyze/sentiment", 
                               json={"text": text})
        print(f"✅ Sentiment Analysis: {response.status_code}")
        result = response.json()
        print(f"   Text: {result['text']}")
        print(f"   Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")
        print(f"   Method: {result['method']}")
    except Exception as e:
        print(f"❌ Sentiment Analysis Failed: {e}")
    
    print()
    
    # Test emotion detection
    try:
        text = "I am very happy with the consultation!"
        response = requests.post(f"{base_url}/analyze/emotion", 
                               json={"text": text})
        print(f"✅ Emotion Detection: {response.status_code}")
        result = response.json()
        print(f"   Text: {result['text']}")
        print(f"   Primary Emotion: {result['primary_emotion']}")
        print(f"   Emotion Scores: {result['emotion_scores']}")
    except Exception as e:
        print(f"❌ Emotion Detection Failed: {e}")
    
    print()
    
    # Test summarization
    try:
        text = "मैंने डॉक्टर से सलाह ली थी। The consultation was very helpful and informative. The doctor explained everything clearly and gave good treatment advice. I am satisfied with the service and would recommend it to others."
        response = requests.post(f"{base_url}/analyze/summarize", 
                               json={"text": text, "max_length": 100})
        print(f"✅ Text Summarization: {response.status_code}")
        result = response.json()
        print(f"   Original Length: {len(result['original_text'])} chars")
        print(f"   Summary: {result['summary']}")
        print(f"   Compression Ratio: {result['compression_ratio']:.2f}")
    except Exception as e:
        print(f"❌ Text Summarization Failed: {e}")
    
    print()
    print("🎉 API Testing Complete!")
    print(f"📖 Visit {base_url}/docs for interactive API documentation")

if __name__ == "__main__":
    test_api()