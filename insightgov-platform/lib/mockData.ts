// Mock analysis data for demo mode
export const mockAnalysisData = {
  "results": [
    {
      "id": 1,
      "originalText": "The e-consultation service is very helpful and saves time. Easy to use interface.",
      "sentiment": {
        "label": "positive",
        "confidence": 0.92,
        "scores": { "positive": 0.92, "negative": 0.04, "neutral": 0.04 },
        "method": "XLM-RoBERTa"
      },
      "emotion": {
        "primary_emotion": "joy",
        "confidence": 0.85,
        "emotion_scores": { "joy": 0.85, "trust": 0.12, "neutral": 0.03 },
        "method": "DistilRoBERTa"
      },
      "summary": "User appreciates the helpful and time-saving e-consultation service with easy interface.",
      "language": "English",
      "timestamp": "2025-10-01T10:30:00Z"
    },
    {
      "id": 2,
      "originalText": "बहुत अच्छी सेवा है। डॉक्टर से बात करना आसान हो गया है।",
      "sentiment": {
        "label": "positive",
        "confidence": 0.89,
        "scores": { "positive": 0.89, "negative": 0.05, "neutral": 0.06 },
        "method": "XLM-RoBERTa"
      },
      "emotion": {
        "primary_emotion": "joy",
        "confidence": 0.82,
        "emotion_scores": { "joy": 0.82, "trust": 0.15, "neutral": 0.03 },
        "method": "DistilRoBERTa"
      },
      "summary": "User finds the service very good and appreciates easy communication with doctors.",
      "language": "Hindi",
      "timestamp": "2025-10-01T10:32:00Z"
    },
    {
      "id": 3,
      "originalText": "The appointment booking system is confusing and takes too long to get a response.",
      "sentiment": {
        "label": "negative",
        "confidence": 0.87,
        "scores": { "positive": 0.08, "negative": 0.87, "neutral": 0.05 },
        "method": "XLM-RoBERTa"
      },
      "emotion": {
        "primary_emotion": "anger",
        "confidence": 0.78,
        "emotion_scores": { "anger": 0.78, "sadness": 0.15, "neutral": 0.07 },
        "method": "DistilRoBERTa"
      },
      "summary": "User frustrated with confusing booking system and slow response times.",
      "language": "English",
      "timestamp": "2025-10-01T10:35:00Z"
    },
    {
      "id": 4,
      "originalText": "System works fine but video quality could be better during consultations.",
      "sentiment": {
        "label": "neutral",
        "confidence": 0.75,
        "scores": { "positive": 0.35, "negative": 0.25, "neutral": 0.40 },
        "method": "XLM-RoBERTa"
      },
      "emotion": {
        "primary_emotion": "neutral",
        "confidence": 0.68,
        "emotion_scores": { "neutral": 0.68, "sadness": 0.22, "joy": 0.10 },
        "method": "DistilRoBERTa"
      },
      "summary": "User satisfied with system functionality but suggests video quality improvements.",
      "language": "English",
      "timestamp": "2025-10-01T10:38:00Z"
    },
    {
      "id": 5,
      "originalText": "डॉक्टर का व्यवहार अच्छा था लेकिन तकनीकी समस्या हुई।",
      "sentiment": {
        "label": "neutral",
        "confidence": 0.72,
        "scores": { "positive": 0.45, "negative": 0.35, "neutral": 0.20 },
        "method": "XLM-RoBERTa"
      },
      "emotion": {
        "primary_emotion": "neutral",
        "confidence": 0.65,
        "emotion_scores": { "neutral": 0.65, "sadness": 0.25, "joy": 0.10 },
        "method": "DistilRoBERTa"
      },
      "summary": "User appreciated doctor's behavior but faced technical issues.",
      "language": "Hindi",
      "timestamp": "2025-10-01T10:40:00Z"
    },
    {
      "id": 6,
      "originalText": "Great service! Quick response time and professional doctors. Highly recommended.",
      "sentiment": {
        "label": "positive",
        "confidence": 0.95,
        "scores": { "positive": 0.95, "negative": 0.02, "neutral": 0.03 },
        "method": "XLM-RoBERTa"
      },
      "emotion": {
        "primary_emotion": "joy",
        "confidence": 0.91,
        "emotion_scores": { "joy": 0.91, "trust": 0.08, "neutral": 0.01 },
        "method": "DistilRoBERTa"
      },
      "summary": "User highly recommends the service praising quick response and professional doctors.",
      "language": "English",
      "timestamp": "2025-10-01T10:42:00Z"
    }
  ],
  "statistics": {
    "total": 6,
    "sentimentCounts": {
      "positive": 3,
      "negative": 1,
      "neutral": 2
    },
    "emotionCounts": {
      "joy": 3,
      "anger": 1,
      "neutral": 2
    },
    "languageCounts": {
      "English": 4,
      "Hindi": 2
    },
    "avgConfidence": 0.835
  },
  "overallSummary": {
    "overall_summary": "Analysis of 6 e-consultation comments reveals 50% positive sentiment with joy being the dominant emotion. The majority of comments are in English with an average confidence of 83.5%. Mixed feedback indicates both satisfaction with professional service and concerns about technical issues.",
    "sentiment_distribution": {
      "positive": 3,
      "negative": 1,
      "neutral": 2
    },
    "emotion_distribution": {
      "joy": 3,
      "anger": 1,
      "neutral": 2
    },
    "key_insights": {
      "satisfaction_level": "Medium",
      "primary_concerns": "Technical Issues"
    },
    "text": "The e-consultation service shows promising user satisfaction with 50% positive feedback. Users appreciate professional doctors and quick response times. However, technical challenges including video quality and system complexity need attention. The service successfully serves both English and Hindi users with high AI confidence scores.",
    "keyInsights": [
      "50% positive sentiment",
      "joy is the primary emotion",
      "English is the main language",
      "84% average confidence"
    ]
  },
  "models_used": {
    "sentiment": "XLM-RoBERTa Multilingual",
    "emotion": "DistilRoBERTa Emotion Classifier",
    "summarization": "DistilBART Summarizer",
    "language_detection": "Fast Heuristic Engine"
  }
}
