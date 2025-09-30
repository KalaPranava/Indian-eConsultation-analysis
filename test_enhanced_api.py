"""
Quick test script to verify all enhanced API endpoints are working
"""
import requests
import json

API_BASE = "http://127.0.0.1:8000"

def test_api():
    print("🧪 Testing Enhanced E-Consultation Analysis API")
    print("=" * 50)
    
    # Test data
    sample_texts = [
        "डॉक्टर बहुत अच्छे हैं। Great service!",
        "The consultation was helpful but waiting time was too long.",
        "बहुत खराब अनुभव था। Very disappointed with the service."
    ]
    
    # Test 1: Health check
    print("\n1. Testing Health Endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"   ✅ Health check: {response.json()}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
    
    # Test 2: Batch Analysis
    print("\n2. Testing Batch Analysis...")
    try:
        response = requests.post(f"{API_BASE}/analyze/batch", 
                               json={"texts": sample_texts, "analyses": ["sentiment", "emotion", "summarize"]})
        data = response.json()
        print(f"   ✅ Batch analysis processed {data['total_processed']} texts")
        for result in data['results'][:2]:  # Show first 2 results
            print(f"      Text {result['id']}: {result['sentiment']['sentiment']} sentiment, {result['emotions']['primary_emotion']} emotion")
    except Exception as e:
        print(f"   ❌ Batch analysis failed: {e}")
    
    # Test 3: Overall Summary
    print("\n3. Testing Overall Summary...")
    try:
        response = requests.post(f"{API_BASE}/analyze/overall_summary", 
                               json={"texts": sample_texts})
        data = response.json()
        print(f"   ✅ Overall summary generated")
        print(f"      Summary: {data['overall_summary'][:100]}...")
        print(f"      Sentiment distribution: {data['sentiment_distribution']}")
    except Exception as e:
        print(f"   ❌ Overall summary failed: {e}")
    
    # Test 4: Word Cloud Data
    print("\n4. Testing Word Cloud...")
    try:
        response = requests.post(f"{API_BASE}/analyze/wordcloud", 
                               json={"texts": sample_texts})
        data = response.json()
        print(f"   ✅ Word cloud data generated")
        print(f"      Found {data['unique_words']} unique words")
        top_words = data['wordcloud_data'][:5]
        print(f"      Top words: {[w['text'] for w in top_words]}")
    except Exception as e:
        print(f"   ❌ Word cloud failed: {e}")
    
    print("\n🎉 API Testing Complete!")
    print("\nNow try uploading sample_econs_comments.csv in the web interface!")

if __name__ == "__main__":
    test_api()