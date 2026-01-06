"""
Health check endpoint for the VQA backend
Add this to your main.py to enable /health endpoint
"""

from main import app

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for deployment"""
    return {
        'status': 'healthy',
        'service': 'vqa-backend'
    }, 200


if __name__ == "__main__":
    app.run(debug=False)
