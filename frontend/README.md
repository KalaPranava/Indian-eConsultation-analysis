# Indian E-Consultation Analysis - Frontend Dashboard

## 🎯 **Complete Frontend Solution**

You now have a **comprehensive web-based dashboard** for analyzing your CSV files of e-consultation comments!

## 🚀 **How to Use**

### **Quick Start:**
1. **Start the API server:**
   ```bash
   # Double-click this file or run in terminal
   start_api.bat
   ```

2. **Start the frontend dashboard:**
   ```bash
   # Double-click this file or run in terminal
   start_frontend.bat
   ```

3. **Open your browser to:** http://localhost:3000

### **Manual Start:**
```bash
# Terminal 1: Start API
cd "d:\SIH\mldl1\New folder"
call venv\Scripts\activate
python -c "import uvicorn; import sys; sys.path.append('scripts'); from working_api import app; uvicorn.run(app, host='127.0.0.1', port=8001)"

# Terminal 2: Start Frontend
cd "d:\SIH\mldl1\New folder\frontend"
python serve_frontend.py
```

## 📊 **Dashboard Features**

### **🔄 CSV Processing:**
- **Drag & Drop** or **Click to Upload** CSV files
- **Real-time validation** of CSV structure
- **Batch processing** with progress tracking
- **Configurable batch size** for optimal performance

### **📈 Analysis Capabilities:**
- **Sentiment Analysis:** Positive/Negative/Neutral detection
- **Emotion Detection:** Joy, Sadness, Anger, Fear, Surprise, Neutral
- **Text Summarization:** Auto-generated summaries for longer comments
- **Language Detection:** Hindi, English, Code-mixed identification

### **📊 Visualizations:**
- **Sentiment Distribution:** Interactive doughnut chart
- **Emotion Analysis:** Bar chart showing emotion frequencies  
- **Confidence Scores:** Line chart of prediction confidence
- **Statistics Cards:** Key metrics at a glance

### **📋 Results Management:**
- **Interactive Data Table:** Sort, search, filter results
- **Detailed View:** Full comment text with analysis results
- **Export Options:** Download as CSV or JSON
- **Summary Insights:** AI-generated recommendations

## 📁 **CSV File Requirements**

### **Required Column:**
- `text` - The comment/feedback text (Hindi/English/Code-mixed)

### **Optional Columns:**
- `comment_id` - Unique identifier
- `user_id` - User identifier
- `rating` - Numerical rating (1-5)  
- `consultation_type` - Type of consultation
- `department` - Medical department
- `timestamp` - When comment was made

### **Example CSV Structure:**
```csv
comment_id,text,user_id,consultation_type,rating
1,"डॉक्टर बहुत अच्छे थे। Very professional service!",user_001,general,5
2,"मुझे परामर्श पसंद नहीं आया। Long wait time.",user_002,specialist,2
3,"यह सेवा उत्कृष्ट है! Highly recommend this platform.",user_003,follow-up,5
```

## 🎨 **Dashboard Sections**

### **1. Upload Section**
- File selection with validation
- Column name configuration
- Batch size settings
- Real-time file information

### **2. Progress Tracking**
- Live progress bar
- Processing status updates
- Estimated time remaining
- Batch-by-batch progress

### **3. Statistics Overview**
- Total comments processed
- Sentiment distribution percentages
- Average confidence scores
- Language detection results
- Top emotions identified

### **4. Interactive Charts**
- **Sentiment Chart:** Visual breakdown of positive/negative/neutral
- **Emotion Chart:** Frequency of different emotions detected
- **Confidence Chart:** Distribution of prediction confidence levels

### **5. Key Insights Summary**
- Automated analysis summary
- Service quality recommendations
- Areas for improvement
- Emotional diversity metrics

### **6. Detailed Results Table**
- Searchable and sortable data grid
- Full comment text with analysis
- Confidence bars for each prediction
- Export functionality for further analysis

## 🔧 **Technical Architecture**

### **Frontend Stack:**
- **HTML5** with Bootstrap 5 for responsive design
- **JavaScript ES6+** with modern async/await
- **Chart.js** for interactive visualizations
- **DataTables** for advanced table functionality
- **Papa Parse** for CSV file processing

### **Backend Integration:**
- **REST API calls** to your FastAPI server
- **Batch processing** for large CSV files
- **Error handling** with graceful fallbacks
- **Real-time progress** updates

### **File Structure:**
```
frontend/
├── index.html              # Main dashboard page
├── js/
│   └── dashboard.js         # Core JavaScript functionality
├── serve_frontend.py        # Python HTTP server
└── README.md               # This documentation
```

## 📊 **Sample Workflow**

1. **Upload CSV:** Drag your consultation comments CSV to the upload area
2. **Configure:** Set text column name and batch size
3. **Process:** Click "Start Analysis" and watch real-time progress
4. **Review:** Examine statistics, charts, and insights
5. **Explore:** Use the interactive table to explore individual results
6. **Export:** Download processed results for further use

## 🎯 **What You'll Get**

### **Immediate Results:**
- Sentiment analysis for every comment
- Emotion detection with confidence scores
- Language identification (Hindi/English/Mixed)
- Text summaries for longer comments

### **Visual Insights:**
- Beautiful charts showing sentiment distribution
- Emotion frequency analysis
- Confidence score patterns
- Statistical overviews

### **Actionable Intelligence:**
- Service quality recommendations
- Areas needing improvement
- Customer satisfaction metrics
- Emotional response patterns

## 🚀 **Getting Started Now**

1. **Place your CSV file** in the `data/` folder or prepare to upload it
2. **Run `start_frontend.bat`** to launch the dashboard
3. **Upload and analyze** your consultation comments
4. **Explore the results** with interactive charts and tables
5. **Export findings** for reports and presentations

Your **complete end-to-end CSV analysis solution** is ready to use! 🎉

## 🔗 **URLs After Starting:**
- **Frontend Dashboard:** http://localhost:3000
- **API Documentation:** http://127.0.0.1:8001/docs
- **API Health Check:** http://127.0.0.1:8001/health