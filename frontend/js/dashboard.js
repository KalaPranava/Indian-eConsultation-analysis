/**
 * Indian E-Consultation Analysis Dashboard
 * JavaScript functionality for CSV upload and analysis
 */

class AnalysisDashboard {
    constructor() {
        this.csvData = null;
        this.analysisResults = [];
        this.apiBaseUrl = 'http://127.0.0.1:8000';
        this.currentBatch = 0;
        this.totalBatches = 0;
        this.startTime = null;
        
        this.initializeEventListeners();
        this.checkApiConnection();
    }

    initializeEventListeners() {
        // File input change event
        document.getElementById('csvFile').addEventListener('change', (e) => {
            this.handleFileSelection(e);
        });

        // Drag and drop events
        const uploadSection = document.querySelector('.upload-section');
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.style.borderColor = '#3498db';
        });

        uploadSection.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadSection.style.borderColor = '#dee2e6';
        });

        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.style.borderColor = '#dee2e6';
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type === 'text/csv') {
                document.getElementById('csvFile').files = files;
                this.handleFileSelection({ target: { files: files } });
            }
        });
    }

    async checkApiConnection() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            if (response.ok) {
                this.showAlert('success', 'API connection successful!', 'Connected to analysis server');
            } else {
                throw new Error('API not responding');
            }
        } catch (error) {
            this.showAlert('danger', 'API Connection Failed', 
                `Cannot connect to analysis server at ${this.apiBaseUrl}. Please ensure the API server is running.`);
        }
    }

    handleFileSelection(event) {
        const file = event.target.files[0];
        if (!file) return;

        const fileInfo = document.getElementById('fileInfo');
        const analyzeBtn = document.getElementById('analyzeBtn');

        if (file.type !== 'text/csv') {
            this.showAlert('danger', 'Invalid File Type', 'Please select a CSV file');
            return;
        }

        // Show file information
        fileInfo.innerHTML = `
            <i class="fas fa-file-csv"></i>
            <strong>File Selected:</strong> ${file.name}<br>
            <strong>Size:</strong> ${this.formatFileSize(file.size)}<br>
            <strong>Last Modified:</strong> ${new Date(file.lastModified).toLocaleString()}
        `;
        fileInfo.style.display = 'block';

        // Parse CSV to validate structure
        Papa.parse(file, {
            header: true,
            preview: 5,
            complete: (results) => {
                this.validateCsvStructure(results.data, results.meta.fields);
                analyzeBtn.disabled = false;
            },
            error: (error) => {
                this.showAlert('danger', 'CSV Parse Error', error.message);
            }
        });
    }

    validateCsvStructure(data, fields) {
        const textColumn = document.getElementById('textColumn').value;
        
        if (!fields.includes(textColumn)) {
            this.showAlert('warning', 'Column Not Found', 
                `Column "${textColumn}" not found. Available columns: ${fields.join(', ')}`);
            return false;
        }

        // Show preview
        const preview = data.slice(0, 3).map(row => row[textColumn]).join(' | ');
        const fileInfo = document.getElementById('fileInfo');
        fileInfo.innerHTML += `<br><strong>Text Preview:</strong> ${preview.substring(0, 200)}...`;
        
        return true;
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    showAlert(type, title, message) {
        const alertHtml = `
            <div class="alert alert-${type} alert-custom alert-dismissible fade show" role="alert">
                <h5><i class="fas fa-${type === 'success' ? 'check-circle' : type === 'danger' ? 'exclamation-triangle' : 'info-circle'}"></i> ${title}</h5>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        // Remove existing alerts
        document.querySelectorAll('.alert-custom').forEach(alert => alert.remove());
        
        // Add new alert
        const container = document.querySelector('.container');
        container.insertAdjacentHTML('afterbegin', alertHtml);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            document.querySelectorAll('.alert-custom').forEach(alert => {
                if (alert.classList.contains('show')) {
                    alert.classList.remove('show');
                    setTimeout(() => alert.remove(), 300);
                }
            });
        }, 5000);
    }

    async startAnalysis() {
        const file = document.getElementById('csvFile').files[0];
        const textColumn = document.getElementById('textColumn').value;
        const batchSize = parseInt(document.getElementById('batchSize').value);

        if (!file) {
            this.showAlert('danger', 'No File Selected', 'Please select a CSV file first');
            return;
        }

        // Reset results
        this.analysisResults = [];
        this.startTime = Date.now();

        // Show progress section
        document.getElementById('progressSection').style.display = 'block';
        document.getElementById('analyzeBtn').disabled = true;

        // Parse full CSV
        Papa.parse(file, {
            header: true,
            complete: async (results) => {
                this.csvData = results.data.filter(row => row[textColumn] && row[textColumn].trim());
                await this.processInBatches(textColumn, batchSize);
            },
            error: (error) => {
                this.showAlert('danger', 'CSV Parse Error', error.message);
                this.resetUI();
            }
        });
    }

    async processInBatches(textColumn, batchSize) {
        const totalComments = this.csvData.length;
        this.totalBatches = Math.ceil(totalComments / batchSize);
        
        document.getElementById('totalComments').textContent = totalComments;

        for (let i = 0; i < totalComments; i += batchSize) {
            const batchEnd = Math.min(i + batchSize, totalComments);
            const batch = this.csvData.slice(i, batchEnd);
            
            this.currentBatch = Math.floor(i / batchSize) + 1;
            
            try {
                await this.processBatch(batch, textColumn, i);
                this.updateProgress(batchEnd, totalComments);
            } catch (error) {
                console.error(`Error processing batch ${this.currentBatch}:`, error);
                this.showAlert('warning', 'Batch Processing Error', 
                    `Error in batch ${this.currentBatch}. Continuing with next batch...`);
            }
        }

        await this.completeAnalysis();
    }

    async processBatch(batch, textColumn, startIndex) {
        const texts = batch.map(row => row[textColumn]);
        
        try {
            // Use the new batch API for better performance
            const batchResponse = await this.callApi('/analyze/batch', {
                texts: texts,
                analyses: ['sentiment', 'emotion', 'summarize']
            });

            // Process batch results
            if (batchResponse && batchResponse.results) {
                for (let i = 0; i < batch.length; i++) {
                    const originalRow = batch[i];
                    const analysisResult = batchResponse.results[i];
                    
                    const result = {
                        id: startIndex + i + 1,
                        originalText: texts[i],
                        originalRow: originalRow,
                        sentiment: analysisResult?.sentiment || this.getFallbackResult('/analyze/sentiment', { text: texts[i] }),
                        emotion: analysisResult?.emotions || this.getFallbackResult('/analyze/emotion', { text: texts[i] }),
                        summary: analysisResult?.summary || texts[i].substring(0, 100) + (texts[i].length > 100 ? '...' : ''),
                        language: this.detectLanguage(texts[i]),
                        timestamp: new Date().toISOString()
                    };
                    
                    this.analysisResults.push(result);
                }
            } else {
                // Fallback to individual processing if batch fails
                await this.processBatchFallback(batch, textColumn, startIndex);
            }

        } catch (error) {
            console.error('Batch processing error, using fallback:', error);
            await this.processBatchFallback(batch, textColumn, startIndex);
        }
    }

    async processBatchFallback(batch, textColumn, startIndex) {
        const texts = batch.map(row => row[textColumn]);
        
        // Process sentiment analysis
        const sentimentPromises = texts.map(text => 
            this.callApi('/analyze/sentiment', { text })
        );
        
        // Process emotion analysis
        const emotionPromises = texts.map(text => 
            this.callApi('/analyze/emotion', { text })
        );
        
        // Process summarization for longer texts
        const summaryPromises = texts.map(text => 
            text.length > 100 ? this.callApi('/analyze/summarize', { text, max_length: 50 }) : Promise.resolve(null)
        );

        // Wait for all analyses to complete
        const [sentimentResults, emotionResults, summaryResults] = await Promise.all([
            Promise.all(sentimentPromises),
            Promise.all(emotionPromises),
            Promise.all(summaryPromises)
        ]);

        // Combine results
        for (let i = 0; i < batch.length; i++) {
            const originalRow = batch[i];
            const result = {
                id: startIndex + i + 1,
                originalText: texts[i],
                originalRow: originalRow,
                sentiment: sentimentResults[i],
                emotion: emotionResults[i],
                summary: summaryResults[i],
                language: this.detectLanguage(texts[i]),
                timestamp: new Date().toISOString()
            };
            
            this.analysisResults.push(result);
        }
    }

    async callApi(endpoint, data) {
        try {
            const response = await fetch(`${this.apiBaseUrl}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error(`API call failed: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API call error:', error);
            // Return fallback result
            return this.getFallbackResult(endpoint, data);
        }
    }

    getFallbackResult(endpoint, data) {
        const text = data.text.toLowerCase();
        
        if (endpoint.includes('sentiment')) {
            const positive = ['good', 'excellent', 'great', 'अच्छा', 'बहुत', 'संतुष्ट'].some(word => text.includes(word));
            const negative = ['bad', 'poor', 'terrible', 'खराब', 'गलत', 'निराश'].some(word => text.includes(word));
            
            return {
                label: positive ? 'positive' : negative ? 'negative' : 'neutral',
                confidence: positive || negative ? 0.7 : 0.5,
                method: 'fallback'
            };
        } else if (endpoint.includes('emotion')) {
            const emotions = {
                joy: ['happy', 'खुश', 'प्रसन्न'].some(word => text.includes(word)) ? 0.7 : 0.1,
                sadness: ['sad', 'दुखी', 'उदास'].some(word => text.includes(word)) ? 0.7 : 0.1,
                anger: ['angry', 'गुस्सा', 'क्रोध'].some(word => text.includes(word)) ? 0.7 : 0.1,
                fear: ['scared', 'डर', 'चिंता'].some(word => text.includes(word)) ? 0.7 : 0.1,
                surprise: ['surprised', 'आश्चर्य'].some(word => text.includes(word)) ? 0.7 : 0.1,
                neutral: 0.3
            };
            
            const primaryEmotion = Object.keys(emotions).reduce((a, b) => emotions[a] > emotions[b] ? a : b);
            
            return {
                primary_emotion: primaryEmotion,
                emotion_scores: emotions,
                method: 'fallback'
            };
        } else if (endpoint.includes('summarize')) {
            const sentences = data.text.split('.').slice(0, 2).join('.') + '.';
            return {
                summary: sentences.length > 100 ? sentences.substring(0, 100) + '...' : sentences,
                method: 'fallback'
            };
        }
        
        return { error: 'Unknown endpoint' };
    }

    detectLanguage(text) {
        const hindiChars = (text.match(/[\u0900-\u097F]/g) || []).length;
        const englishChars = (text.match(/[a-zA-Z]/g) || []).length;
        const total = hindiChars + englishChars;
        
        if (total === 0) return 'unknown';
        
        const hindiRatio = hindiChars / total;
        const englishRatio = englishChars / total;
        
        if (hindiRatio > 0.6) return 'Hindi';
        if (englishRatio > 0.6) return 'English';
        if (hindiRatio > 0.2 && englishRatio > 0.2) return 'Code-mixed';
        
        return 'Mixed';
    }

    updateProgress(processed, total) {
        const percentage = Math.round((processed / total) * 100);
        const progressBar = document.getElementById('progressBar');
        const currentComment = document.getElementById('currentComment');
        const timeRemaining = document.getElementById('timeRemaining');
        
        progressBar.style.width = `${percentage}%`;
        progressBar.textContent = `${percentage}%`;
        currentComment.textContent = processed;
        
        // Calculate estimated time remaining
        const elapsed = (Date.now() - this.startTime) / 1000;
        const rate = processed / elapsed;
        const remaining = total - processed;
        const eta = remaining / rate;
        
        if (eta > 60) {
            timeRemaining.textContent = `${Math.round(eta / 60)} minutes`;
        } else {
            timeRemaining.textContent = `${Math.round(eta)} seconds`;
        }
    }

    async completeAnalysis() {
        // Hide progress, show results
        document.getElementById('progressSection').style.display = 'none';
        
        this.generateStatistics();
        this.generateCharts();
        this.generateSummary();
        this.populateResultsTable();
        
        // Show all result sections
        document.getElementById('statsSection').style.display = 'block';
        document.getElementById('chartsSection').style.display = 'block';
        document.getElementById('summarySection').style.display = 'block';
        document.getElementById('resultsSection').style.display = 'block';
        
        // Re-enable analyze button
        document.getElementById('analyzeBtn').disabled = false;
        
        this.showAlert('success', 'Analysis Complete!', 
            `Successfully analyzed ${this.analysisResults.length} comments. Check the results below.`);
    }

    generateStatistics() {
        const results = this.analysisResults;
        const total = results.length;
        
        // Sentiment statistics
        const sentimentCounts = {
            positive: results.filter(r => r.sentiment?.label === 'positive').length,
            negative: results.filter(r => r.sentiment?.label === 'negative').length,
            neutral: results.filter(r => r.sentiment?.label === 'neutral').length
        };
        
        // Emotion statistics
        const emotions = results.map(r => r.emotion?.primary_emotion).filter(Boolean);
        const emotionCounts = emotions.reduce((acc, emotion) => {
            acc[emotion] = (acc[emotion] || 0) + 1;
            return acc;
        }, {});
        
        // Language statistics
        const languages = results.map(r => r.language).filter(Boolean);
        const languageCounts = languages.reduce((acc, lang) => {
            acc[lang] = (acc[lang] || 0) + 1;
            return acc;
        }, {});
        
        // Average confidence
        const avgConfidence = results
            .map(r => r.sentiment?.confidence || 0)
            .reduce((a, b) => a + b, 0) / total;

        this.renderStatsCards({
            total,
            sentimentCounts,
            emotionCounts,
            languageCounts,
            avgConfidence
        });
    }

    renderStatsCards(stats) {
        const statsContainer = document.getElementById('statsCards');
        
        const cards = [
            {
                title: 'Total Comments',
                value: stats.total,
                icon: 'fas fa-comments',
                color: 'primary'
            },
            {
                title: 'Positive Sentiment',
                value: `${stats.sentimentCounts.positive} (${Math.round(stats.sentimentCounts.positive/stats.total*100)}%)`,
                icon: 'fas fa-smile',
                color: 'success'
            },
            {
                title: 'Negative Sentiment',
                value: `${stats.sentimentCounts.negative} (${Math.round(stats.sentimentCounts.negative/stats.total*100)}%)`,
                icon: 'fas fa-frown',
                color: 'danger'
            },
            {
                title: 'Average Confidence',
                value: `${(stats.avgConfidence * 100).toFixed(1)}%`,
                icon: 'fas fa-chart-line',
                color: 'info'
            },
            {
                title: 'Languages Detected',
                value: Object.keys(stats.languageCounts).length,
                icon: 'fas fa-language',
                color: 'warning'
            },
            {
                title: 'Top Emotion',
                value: Object.keys(stats.emotionCounts).reduce((a, b) => 
                    stats.emotionCounts[a] > stats.emotionCounts[b] ? a : b, 'none'),
                icon: 'fas fa-heart',
                color: 'secondary'
            }
        ];

        statsContainer.innerHTML = cards.map(card => `
            <div class="col-md-4 col-sm-6">
                <div class="stats-card">
                    <div class="stats-icon text-${card.color}">
                        <i class="${card.icon}"></i>
                    </div>
                    <h3 class="text-${card.color}">${card.value}</h3>
                    <p class="text-muted mb-0">${card.title}</p>
                </div>
            </div>
        `).join('');
    }

    generateCharts() {
        this.createSentimentChart();
        this.createEmotionChart();
        this.createConfidenceChart();
    }

    createSentimentChart() {
        const ctx = document.getElementById('sentimentChart').getContext('2d');
        const results = this.analysisResults;
        
        const sentimentCounts = {
            positive: results.filter(r => r.sentiment?.label === 'positive').length,
            negative: results.filter(r => r.sentiment?.label === 'negative').length,
            neutral: results.filter(r => r.sentiment?.label === 'neutral').length
        };

        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Negative', 'Neutral'],
                datasets: [{
                    data: [sentimentCounts.positive, sentimentCounts.negative, sentimentCounts.neutral],
                    backgroundColor: ['#27ae60', '#e74c3c', '#f39c12'],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const percentage = Math.round((context.parsed / results.length) * 100);
                                return `${context.label}: ${context.parsed} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    createEmotionChart() {
        const ctx = document.getElementById('emotionChart').getContext('2d');
        const results = this.analysisResults;
        
        const emotions = results.map(r => r.emotion?.primary_emotion).filter(Boolean);
        const emotionCounts = emotions.reduce((acc, emotion) => {
            acc[emotion] = (acc[emotion] || 0) + 1;
            return acc;
        }, {});

        const colors = {
            joy: '#f1c40f',
            sadness: '#3498db',
            anger: '#e74c3c',
            fear: '#9b59b6',
            surprise: '#e67e22',
            neutral: '#95a5a6'
        };

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Object.keys(emotionCounts),
                datasets: [{
                    label: 'Count',
                    data: Object.values(emotionCounts),
                    backgroundColor: Object.keys(emotionCounts).map(emotion => colors[emotion] || '#95a5a6'),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    createConfidenceChart() {
        const ctx = document.getElementById('confidenceChart').getContext('2d');
        const results = this.analysisResults;
        
        const confidenceRanges = {
            'Very High (>90%)': 0,
            'High (70-90%)': 0,
            'Medium (50-70%)': 0,
            'Low (<50%)': 0
        };

        results.forEach(result => {
            const confidence = result.sentiment?.confidence || 0;
            if (confidence > 0.9) confidenceRanges['Very High (>90%)']++;
            else if (confidence > 0.7) confidenceRanges['High (70-90%)']++;
            else if (confidence > 0.5) confidenceRanges['Medium (50-70%)']++;
            else confidenceRanges['Low (<50%)']++;
        });

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: Object.keys(confidenceRanges),
                datasets: [{
                    label: 'Number of Comments',
                    data: Object.values(confidenceRanges),
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });
    }

    async generateSummary() {
        const results = this.analysisResults;
        const texts = results.map(r => r.originalText);
        
        try {
            // Get AI-generated overall summary
            const overallSummaryResponse = await this.callApi('/analyze/overall_summary', {
                texts: texts,
                max_length: 200
            });
            
            // Get word cloud data
            const wordcloudResponse = await this.callApi('/analyze/wordcloud', {
                texts: texts,
                analyses: []
            });
            
            this.renderAISummary(overallSummaryResponse, wordcloudResponse);
            
        } catch (error) {
            console.error('Error generating AI summary:', error);
            // Fallback to basic summary
            this.generateBasicSummary();
        }
    }

    renderAISummary(summaryData, wordcloudData) {
        const summaryContent = `
            <div class="row">
                <div class="col-12">
                    <h6><i class="fas fa-robot"></i> AI-Generated Summary</h6>
                    <div class="alert alert-info">
                        <strong>Overall Analysis:</strong> ${summaryData.overall_summary}
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-4">
                    <h6><i class="fas fa-chart-pie"></i> Sentiment Distribution</h6>
                    <ul class="list-unstyled">
                        <li>• Positive: ${summaryData.sentiment_distribution?.positive || 0} comments</li>
                        <li>• Negative: ${summaryData.sentiment_distribution?.negative || 0} comments</li>
                        <li>• Neutral: ${summaryData.sentiment_distribution?.neutral || 0} comments</li>
                    </ul>
                </div>
                <div class="col-md-4">
                    <h6><i class="fas fa-heart"></i> Emotion Distribution</h6>
                    <ul class="list-unstyled">
                        ${Object.entries(summaryData.emotion_distribution || {})
                            .sort(([,a], [,b]) => b - a)
                            .slice(0, 3)
                            .map(([emotion, count]) => `<li>• ${emotion}: ${count} comments</li>`)
                            .join('')}
                    </ul>
                </div>
                <div class="col-md-4">
                    <h6><i class="fas fa-lightbulb"></i> Key Insights</h6>
                    <ul class="list-unstyled">
                        <li>• Satisfaction: ${summaryData.key_insights?.satisfaction_level || 'moderate'}</li>
                        <li>• Primary concerns: ${summaryData.key_insights?.primary_concerns || 'service quality'}</li>
                        <li>• Emotional state: ${summaryData.key_insights?.emotional_state || 'neutral'}</li>
                    </ul>
                </div>
            </div>
            
            <div class="mt-3">
                <h6><i class="fas fa-cloud"></i> Interactive Word Cloud</h6>
                <div id="wordCloudContainer" class="border rounded p-3" style="min-height: 300px; background: #f8f9fa;">
                    <div class="text-center">
                        <div class="spinner-border" role="status">
                            <span class="visually-hidden">Loading word cloud...</span>
                        </div>
                    </div>
                </div>
                <small class="text-muted">Click on words to see comments containing them</small>
            </div>
            
            <div class="mt-3" id="wordClickResults" style="display: none;">
                <h6><i class="fas fa-search"></i> Comments containing "<span id="selectedWord"></span>"</h6>
                <div id="selectedWordComments" class="border rounded p-3 max-height-300 overflow-auto">
                </div>
            </div>
        `;
        
        document.getElementById('summaryContent').innerHTML = summaryContent;
        
        // Generate interactive word cloud
        this.generateWordCloud(wordcloudData);
    }

    generateBasicSummary() {
        const results = this.analysisResults;
        const total = results.length;
        
        const positivePercentage = Math.round((results.filter(r => r.sentiment?.label === 'positive').length / total) * 100);
        const negativePercentage = Math.round((results.filter(r => r.sentiment?.label === 'negative').length / total) * 100);
        
        const topEmotion = results
            .map(r => r.emotion?.primary_emotion)
            .filter(Boolean)
            .reduce((acc, emotion) => {
                acc[emotion] = (acc[emotion] || 0) + 1;
                return acc;
            }, {});
        
        const mostCommonEmotion = Object.keys(topEmotion).reduce((a, b) => 
            topEmotion[a] > topEmotion[b] ? a : b, 'none');
        
        const avgConfidence = Math.round((results.reduce((sum, r) => sum + (r.sentiment?.confidence || 0), 0) / total) * 100);
        
        const summaryContent = `
            <div class="row">
                <div class="col-md-6">
                    <h6><i class="fas fa-chart-pie"></i> Sentiment Overview</h6>
                    <ul class="list-unstyled">
                        <li>• ${positivePercentage}% of comments express positive sentiment</li>
                        <li>• ${negativePercentage}% of comments express negative sentiment</li>
                        <li>• Average confidence score: ${avgConfidence}%</li>
                    </ul>
                </div>
                <div class="col-md-6">
                    <h6><i class="fas fa-heart"></i> Emotion Insights</h6>
                    <ul class="list-unstyled">
                        <li>• Most common emotion: ${mostCommonEmotion}</li>
                        <li>• ${Object.keys(topEmotion).length} different emotions detected</li>
                        <li>• Emotional diversity score: ${Math.round((Object.keys(topEmotion).length / 6) * 100)}%</li>
                    </ul>
                </div>
            </div>
        `;
        
        document.getElementById('summaryContent').innerHTML = summaryContent;
    }

    generateWordCloud(wordcloudData) {
        if (!wordcloudData || !wordcloudData.wordcloud_data) {
            document.getElementById('wordCloudContainer').innerHTML = '<p class="text-muted text-center">Word cloud data not available</p>';
            return;
        }

        const container = document.getElementById('wordCloudContainer');
        container.innerHTML = '';
        container.style.textAlign = 'center';
        container.style.lineHeight = '1.8';
        container.style.padding = '20px';
        container.style.background = 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)';
        container.style.borderRadius = '15px';
        container.style.boxShadow = 'inset 0 2px 10px rgba(0,0,0,0.1)';
        
        // Create word cloud using enhanced HTML/CSS
        const words = wordcloudData.wordcloud_data.slice(0, 40); // Top 40 words
        const maxSize = Math.max(...words.map(w => w.size));
        const minSize = Math.min(...words.map(w => w.size));
        
        // Color palettes for different frequency ranges
        const colorPalettes = [
            ['#e74c3c', '#c0392b'], // High frequency - Red
            ['#3498db', '#2980b9'], // Medium-high - Blue  
            ['#2ecc71', '#27ae60'], // Medium - Green
            ['#f39c12', '#e67e22'], // Medium-low - Orange
            ['#9b59b6', '#8e44ad'], // Low - Purple
            ['#34495e', '#2c3e50']  // Very low - Gray
        ];
        
        words.forEach((wordData, index) => {
            const span = document.createElement('span');
            
            // Calculate size based on frequency (12px to 36px)
            const sizeRatio = (wordData.size - minSize) / Math.max(maxSize - minSize, 1);
            const fontSize = Math.max(14, Math.min(36, 14 + sizeRatio * 22));
            
            span.textContent = wordData.text;
            span.style.fontSize = `${fontSize}px`;
            span.style.fontWeight = sizeRatio > 0.7 ? 'bold' : sizeRatio > 0.4 ? '600' : '500';
            span.style.margin = '8px 12px';
            span.style.display = 'inline-block';
            span.style.cursor = 'pointer';
            span.style.padding = '8px 12px';
            span.style.borderRadius = '25px';
            span.style.transition = 'all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
            span.className = 'word-cloud-item';
            span.title = `"${wordData.text}" appears in ${wordData.size} comments - Click to view`;
            
            // Enhanced color scheme based on frequency
            const paletteIndex = Math.floor((1 - sizeRatio) * (colorPalettes.length - 1));
            const palette = colorPalettes[paletteIndex];
            const baseColor = palette[0];
            const darkColor = palette[1];
            
            span.style.background = `linear-gradient(135deg, ${baseColor}, ${darkColor})`;
            span.style.color = 'white';
            span.style.textShadow = '1px 1px 2px rgba(0,0,0,0.3)';
            span.style.boxShadow = '0 4px 15px rgba(0,0,0,0.2)';
            span.style.border = '2px solid rgba(255,255,255,0.3)';
            
            // Enhanced click handler
            span.addEventListener('click', () => {
                this.showWordComments(wordData.text, wordData.comments);
                
                // Highlight selected word with animation
                document.querySelectorAll('.word-cloud-item').forEach(item => {
                    item.style.transform = 'scale(1)';
                    item.style.boxShadow = '0 4px 15px rgba(0,0,0,0.2)';
                    item.style.zIndex = '1';
                });
                
                span.style.transform = 'scale(1.2)';
                span.style.boxShadow = '0 8px 25px rgba(0,0,0,0.4)';
                span.style.zIndex = '10';
                
                // Add pulse animation
                span.style.animation = 'pulse 0.6s ease-in-out';
            });
            
            // Enhanced hover effects
            span.addEventListener('mouseenter', () => {
                if (span.style.zIndex !== '10') {
                    span.style.transform = 'scale(1.15) translateY(-2px)';
                    span.style.boxShadow = '0 6px 20px rgba(0,0,0,0.3)';
                    span.style.zIndex = '5';
                }
            });
            
            span.addEventListener('mouseleave', () => {
                if (span.style.zIndex !== '10') {
                    span.style.transform = 'scale(1) translateY(0)';
                    span.style.boxShadow = '0 4px 15px rgba(0,0,0,0.2)';
                    span.style.zIndex = '1';
                }
            });
            
            // Random positioning for more natural cloud effect
            const delay = Math.random() * 0.5;
            span.style.animationDelay = `${delay}s`;
            
            container.appendChild(span);
        });
        
        // Add CSS animations
        const style = document.createElement('style');
        style.textContent = `
            @keyframes pulse {
                0%, 100% { transform: scale(1.2); }
                50% { transform: scale(1.3); }
            }
            
            .word-cloud-item {
                animation: fadeInUp 0.6s ease-out forwards;
                opacity: 0;
                transform: translateY(20px);
            }
            
            @keyframes fadeInUp {
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
        `;
        document.head.appendChild(style);
        
        // Add word cloud statistics
        const stats = document.createElement('div');
        stats.className = 'mt-3 text-center';
        stats.innerHTML = `
            <small class="text-muted">
                <i class="fas fa-info-circle"></i> 
                Showing top ${words.length} words from ${wordcloudData.total_unique_words} unique words
                <br>
                <span class="badge bg-primary me-2">Large = Frequent</span>
                <span class="badge bg-success me-2">Click to explore</span>
                <span class="badge bg-info">Hover for details</span>
            </small>
        `;
        container.appendChild(stats);
    }

    showWordComments(word, comments) {
        document.getElementById('selectedWord').textContent = word;
        document.getElementById('wordClickResults').style.display = 'block';
        
        const commentsContainer = document.getElementById('selectedWordComments');
        
        if (!comments || comments.length === 0) {
            commentsContainer.innerHTML = '<p class="text-muted">No comments found for this word.</p>';
            return;
        }
        
        const commentsHtml = comments.map(comment => `
            <div class="border-bottom pb-2 mb-2">
                <small class="text-muted">Comment #${comment.id}</small>
                <p class="mb-0">${comment.text}</p>
            </div>
        `).join('');
        
        commentsContainer.innerHTML = commentsHtml;
        
        // Scroll to results
        document.getElementById('wordClickResults').scrollIntoView({ behavior: 'smooth' });
    }

    populateResultsTable() {
        const tableBody = document.getElementById('resultsTableBody');
        
        const rows = this.analysisResults.map(result => {
            const sentiment = result.sentiment || {};
            const emotion = result.emotion || {};
            const summary = result.summary || {};
            
            const emotionScoresHtml = Object.entries(emotion.emotion_scores || {})
                .sort(([,a], [,b]) => b - a)
                .slice(0, 3)
                .map(([em, score]) => `<span class="badge bg-secondary me-1">${em}: ${(score * 100).toFixed(0)}%</span>`)
                .join('');
            
            const confidenceBarWidth = (sentiment.confidence || 0) * 100;
            const confidenceColor = confidenceBarWidth > 70 ? '#27ae60' : confidenceBarWidth > 50 ? '#f39c12' : '#e74c3c';
            
            return `
                <tr>
                    <td>${result.id}</td>
                    <td class="comment-cell">${this.truncateText(result.originalText, 100)}</td>
                    <td>
                        <span class="badge bg-${sentiment.label === 'positive' ? 'success' : sentiment.label === 'negative' ? 'danger' : 'warning'}">
                            ${sentiment.label || 'Unknown'}
                        </span>
                    </td>
                    <td>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidenceBarWidth}%; background-color: ${confidenceColor}"></div>
                        </div>
                        <small>${(sentiment.confidence * 100 || 0).toFixed(1)}%</small>
                    </td>
                    <td>
                        <span class="badge bg-info">${emotion.primary_emotion || 'Unknown'}</span>
                    </td>
                    <td>${emotionScoresHtml}</td>
                    <td class="comment-cell">${this.truncateText(result.summary || 'N/A', 80)}</td>
                    <td>
                        <span class="badge bg-secondary">${result.language}</span>
                    </td>
                </tr>
            `;
        }).join('');
        
        tableBody.innerHTML = rows;
        
        // Initialize DataTables
        if ($.fn.DataTable.isDataTable('#resultsTable')) {
            $('#resultsTable').DataTable().destroy();
        }
        
        $('#resultsTable').DataTable({
            pageLength: 25,
            responsive: true,
            order: [[0, 'asc']],
            columnDefs: [
                { orderable: false, targets: [5] },
                { searchable: true, targets: [1, 2, 4, 6, 7] }
            ],
            language: {
                search: "Search comments:",
                lengthMenu: "Show _MENU_ comments per page",
                info: "Showing _START_ to _END_ of _TOTAL_ comments",
                paginate: {
                    first: "First",
                    last: "Last",
                    next: "Next",
                    previous: "Previous"
                }
            }
        });
    }

    truncateText(text, maxLength) {
        if (!text) return 'N/A';
        return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
    }

    exportToCSV() {
        const csvContent = [
            ['ID', 'Original Text', 'Sentiment', 'Confidence', 'Primary Emotion', 'Emotion Scores', 'Summary', 'Language', 'Timestamp'],
            ...this.analysisResults.map(result => [
                result.id,
                `"${result.originalText.replace(/"/g, '""')}"`,
                result.sentiment?.label || '',
                result.sentiment?.confidence || '',
                result.emotion?.primary_emotion || '',
                JSON.stringify(result.emotion?.emotion_scores || {}),
                `"${(result.summary?.summary || '').replace(/"/g, '""')}"`,
                result.language,
                result.timestamp
            ])
        ].map(row => row.join(',')).join('\n');

        this.downloadFile(csvContent, 'analysis_results.csv', 'text/csv');
    }

    exportToJSON() {
        const jsonContent = JSON.stringify(this.analysisResults, null, 2);
        this.downloadFile(jsonContent, 'analysis_results.json', 'application/json');
    }

    downloadFile(content, filename, contentType) {
        const blob = new Blob([content], { type: contentType });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }

    resetUI() {
        document.getElementById('progressSection').style.display = 'none';
        document.getElementById('statsSection').style.display = 'none';
        document.getElementById('chartsSection').style.display = 'none';
        document.getElementById('summarySection').style.display = 'none';
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = false;
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new AnalysisDashboard();
});

// Global functions for button events
function startAnalysis() {
    window.dashboard.startAnalysis();
}

function exportToCSV() {
    window.dashboard.exportToCSV();
}

function exportToJSON() {
    window.dashboard.exportToJSON();
}