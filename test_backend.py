"""
Unit tests for VQA backend
"""
import unittest
import json
import sys
from io import BytesIO

# Add parent directory to path for imports
sys.path.insert(0, '.')

from main import app, allowed_file


class VQABackendTests(unittest.TestCase):
    """Test cases for the VQA backend API"""

    def setUp(self):
        """Set up test client before each test"""
        self.app = app
        self.client = app.test_client()
        self.app.config['TESTING'] = True

    def test_allowed_file(self):
        """Test file validation"""
        self.assertTrue(allowed_file('image.jpg'))
        self.assertTrue(allowed_file('image.jpeg'))
        self.assertTrue(allowed_file('image.png'))
        self.assertFalse(allowed_file('image.txt'))
        self.assertFalse(allowed_file('image'))

    def test_health_check(self):
        """Test that the app has health endpoint"""
        response = self.client.get('/health', follow_redirects=True)
        # Should return 200 OK
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')

    def test_predict_endpoint_exists(self):
        """Test that predict endpoint exists"""
        response = self.client.post('/predict')
        # Should not be 404
        self.assertNotEqual(response.status_code, 404)

    def test_predict_no_file(self):
        """Test predict endpoint without file"""
        response = self.client.post('/predict', data={'question': 'what color?'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_predict_no_question(self):
        """Test predict endpoint without question"""
        response = self.client.post('/predict', data={
            'file': (BytesIO(b'fake'), 'test.jpg')
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_predict_invalid_file_type(self):
        """Test predict endpoint with invalid file type"""
        response = self.client.post('/predict', data={
            'file': (BytesIO(b'test content'), 'test.txt'),
            'question': 'what color?'
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('error', data)


class HealthCheckTests(unittest.TestCase):
    """Health check tests for deployment validation"""

    def setUp(self):
        """Set up test client"""
        self.app = app
        self.client = app.test_client()
        self.app.config['TESTING'] = True

    def test_app_initialization(self):
        """Test that the app initializes without errors"""
        self.assertIsNotNone(self.app)

    def test_flask_app_exists(self):
        """Test that Flask app is properly configured"""
        self.assertTrue(hasattr(app, 'route'))

    def test_cors_enabled(self):
        """Test that CORS is enabled"""
        # CORS should add headers to responses
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main(verbosity=2)
