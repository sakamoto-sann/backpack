#!/usr/bin/env python3
"""
Flask web API wrapper for URL summarizer
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from url_summarizer_api import summarize_url

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "service": "URL Summarizer API",
        "status": "healthy",
        "usage": "POST /summarize with JSON body: {\"url\": \"https://example.com\"}"
    })

@app.route('/summarize', methods=['POST'])
def summarize():
    """Main summarization endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({
                "error": "Missing 'url' parameter in JSON body",
                "status": "error"
            }), 400
        
        url = data['url']
        if not url or not isinstance(url, str):
            return jsonify({
                "error": "URL must be a non-empty string",
                "status": "error"
            }), 400
        
        result = summarize_url(url)
        
        if result['status'] == 'error':
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": f"Server error: {str(e)}",
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check for load balancers"""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)