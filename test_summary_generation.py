"""
Test script to verify summary generation is working
Run this after starting the API with start_working_ml_enhanced.bat
"""
import requests
import json

def test_summary_generation():
    api_url = "http://localhost:8000"
    
    # Test data - sample e-consultation comments
    test_comments = [
        "I think this policy is very good for our community. It will help many people.",
        "The implementation seems rushed. We need more time to understand all implications.",
        "This policy addresses important issues but needs better funding allocation.",
        "I support the overall direction but have concerns about the timeline.",
        "Great initiative! This will definitely improve the situation for citizens."
    ]
    
    print("🧪 Testing Summary Generation...")
    print("=" * 50)
    
    # Test individual summary
    print("\n1. Testing Individual Comment Summary:")
    try:
        response = requests.post(f"{api_url}/analyze/summarize", 
                               json={"text": test_comments[0], "max_summary_length": 50})
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Original: {test_comments[0]}")
            print(f"✅ Summary: {result.get('summary', 'N/A')}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    # Test overall summary generation
    print("\n2. Testing Overall Summary Generation:")
    try:
        response = requests.post(f"{api_url}/analyze/overall_summary", 
                               json={"comments": test_comments})
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Overall Summary: {result.get('overall_summary', 'N/A')}")
            if 'abstractive_summary' in result and result['abstractive_summary']:
                print(f"✅ Abstractive Summary: {result['abstractive_summary']}")
            else:
                print("⚠️ No abstractive summary generated")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    # Test API health
    print("\n3. Testing API Health:")
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ API Status: {health.get('status', 'Unknown')}")
            print(f"✅ Models Loaded: {health.get('models_loaded', 'Unknown')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")

if __name__ == "__main__":
    print("Make sure the API is running with start_working_ml_enhanced.bat")
    input("Press Enter to start testing...")
    test_summary_generation()
    input("\nPress Enter to exit...")