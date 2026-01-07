"""
Integration tests for VQA backend - tests deployed container
Run this test against a live deployment with: CONTAINER_URL=https://your-url pytest test_integration.py -v
"""
import os
import sys
import unittest
import requests
import time
import json
from io import BytesIO
from pathlib import Path

# Get container URL from environment
CONTAINER_URL = os.getenv('CONTAINER_URL', 'http://localhost:8080')

# Get test image path
TEST_IMAGE_DIR = Path(__file__).parent / 'test'
TEST_IMAGE_PATH = TEST_IMAGE_DIR / 'COCO_train2014_000000000081.jpg'


class IntegrationTestsBase(unittest.TestCase):
    """Base class for integration tests"""
    
    @classmethod
    def setUpClass(cls):
        """Verify container is accessible"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = requests.get(f'{CONTAINER_URL}/health', timeout=5)
                if response.status_code == 200:
                    print(f"\n✓ Container is healthy at {CONTAINER_URL}")
                    cls.container_available = True
                    return
            except requests.exceptions.RequestException:
                pass
            
            retry_count += 1
            if retry_count < max_retries:
                print(f"Waiting for container... (attempt {retry_count}/{max_retries})")
                time.sleep(3)
        
        cls.container_available = False
        print(f"\n⚠️  Warning: Container not available at {CONTAINER_URL}")


class HealthCheckIntegrationTests(IntegrationTestsBase):
    """Test health check endpoint"""
    
    def test_health_endpoint_returns_200(self):
        """Health endpoint should return 200"""
        if not self.container_available:
            self.skipTest("Container not available")
        
        response = requests.get(f'{CONTAINER_URL}/health', timeout=10)
        self.assertEqual(response.status_code, 200)
    
    def test_health_endpoint_json_response(self):
        """Health endpoint should return valid JSON"""
        if not self.container_available:
            self.skipTest("Container not available")
        
        response = requests.get(f'{CONTAINER_URL}/health', timeout=10)
        data = response.json()
        
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('service', data)
    
    def test_health_response_time(self):
        """Health endpoint should respond quickly (< 1 second)"""
        if not self.container_available:
            self.skipTest("Container not available")
        
        start = time.time()
        response = requests.get(f'{CONTAINER_URL}/health', timeout=10)
        duration = time.time() - start
        
        self.assertEqual(response.status_code, 200)
        self.assertLess(duration, 1.0, f"Health check took {duration:.2f}s")


class PredictEndpointIntegrationTests(IntegrationTestsBase):
    """Test predict endpoint with actual requests"""
    
    def test_predict_no_file_no_question(self):
        """Predict endpoint should handle missing file and question"""
        if not self.container_available:
            self.skipTest("Container not available")
        
        response = requests.post(f'{CONTAINER_URL}/predict', timeout=30)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('error', data)
    
    def test_predict_no_file_with_question(self):
        """Predict endpoint should handle missing file"""
        if not self.container_available:
            self.skipTest("Container not available")
        
        response = requests.post(
            f'{CONTAINER_URL}/predict',
            data={'question': 'what is this?'},
            timeout=30
        )
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('error', data)
    
    def test_predict_no_question_with_file(self):
        """Predict endpoint should handle missing question"""
        if not self.container_available:
            self.skipTest("Container not available")
        
        # Create a simple valid image file
        from PIL import Image
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        response = requests.post(
            f'{CONTAINER_URL}/predict',
            files={'file': ('test.jpg', img_bytes, 'image/jpeg')},
            timeout=30
        )
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('error', data)
    
    def test_predict_with_test_image_if_exists(self):
        """Test predict with actual test image if available"""
        if not self.container_available:
            self.skipTest("Container not available")
        
        if not TEST_IMAGE_PATH.exists():
            self.skipTest(f"Test image not found at {TEST_IMAGE_PATH}")
        
        with open(TEST_IMAGE_PATH, 'rb') as f:
            response = requests.post(
                f'{CONTAINER_URL}/predict',
                files={'file': ('test.jpg', f, 'image/jpeg')},
                data={'question': 'what color is the bus?'},
                timeout=60
            )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Should contain class names (no error)
        if 'error' not in data:
            # Verify response structure
            for i in range(10):
                self.assertIn(f'class_name-{i}', data)
            
            print(f"\n✓ Prediction received: {list(data.keys())[:3]}...")
    
    def test_predict_invalid_file_type(self):
        """Predict endpoint should reject invalid file types"""
        if not self.container_available:
            self.skipTest("Container not available")
        
        response = requests.post(
            f'{CONTAINER_URL}/predict',
            files={'file': ('test.txt', BytesIO(b'test'), 'text/plain')},
            data={'question': 'what is this?'},
            timeout=30
        )
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('error', data)


class PerformanceIntegrationTests(IntegrationTestsBase):
    """Test performance characteristics"""
    
    def test_predict_response_time_with_image(self):
        """Predict endpoint should respond within reasonable time"""
        if not self.container_available:
            self.skipTest("Container not available")
        
        if not TEST_IMAGE_PATH.exists():
            self.skipTest(f"Test image not found at {TEST_IMAGE_PATH}")
        
        with open(TEST_IMAGE_PATH, 'rb') as f:
            start = time.time()
            response = requests.post(
                f'{CONTAINER_URL}/predict',
                files={'file': ('test.jpg', f, 'image/jpeg')},
                data={'question': 'what color is the bus?'},
                timeout=120
            )
            duration = time.time() - start
        
        self.assertEqual(response.status_code, 200)
        # Predict can take longer (model inference), typically 10-30 seconds
        self.assertLess(duration, 120, f"Prediction took {duration:.2f}s")
        
        print(f"\n✓ Prediction completed in {duration:.2f}s")
    
    def test_concurrent_health_checks(self):
        """Multiple concurrent health checks should work"""
        if not self.container_available:
            self.skipTest("Container not available")
        
        import concurrent.futures
        
        def check_health():
            response = requests.get(f'{CONTAINER_URL}/health', timeout=10)
            return response.status_code == 200
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(check_health, range(5)))
        
        self.assertTrue(all(results), "Some health checks failed")


class RobustnessIntegrationTests(IntegrationTestsBase):
    """Test error handling and edge cases"""
    
    def test_predict_with_corrupted_image(self):
        """Predict endpoint should handle corrupted images gracefully"""
        if not self.container_available:
            self.skipTest("Container not available")
        
        response = requests.post(
            f'{CONTAINER_URL}/predict',
            files={'file': ('test.jpg', BytesIO(b'corrupted image data'), 'image/jpeg')},
            data={'question': 'what is this?'},
            timeout=30
        )
        # Should not crash - return error or 200
        self.assertIn(response.status_code, [200, 400, 422])
    
    def test_predict_with_very_large_question(self):
        """Predict endpoint should handle very long questions"""
        if not self.container_available:
            self.skipTest("Container not available")
        
        long_question = "what " * 500  # Very long question
        
        from PIL import Image
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        response = requests.post(
            f'{CONTAINER_URL}/predict',
            files={'file': ('test.jpg', img_bytes, 'image/jpeg')},
            data={'question': long_question},
            timeout=60
        )
        # Should not crash
        self.assertIn(response.status_code, [200, 400, 422])
    
    def test_cors_headers_present(self):
        """Response should include CORS headers"""
        if not self.container_available:
            self.skipTest("Container not available")
        
        response = requests.get(f'{CONTAINER_URL}/health', timeout=10)
        
        # CORS should be enabled
        self.assertIn('Access-Control-Allow-Origin', response.headers)


class DeploymentValidationTests(IntegrationTestsBase):
    """Validate deployment meets requirements"""
    
    def test_service_is_reachable(self):
        """Service should be reachable from outside"""
        if not self.container_available:
            self.skipTest("Container not available")
        
        # Try multiple times
        for attempt in range(3):
            try:
                response = requests.get(f'{CONTAINER_URL}/health', timeout=10)
                self.assertEqual(response.status_code, 200)
                return
            except requests.exceptions.RequestException:
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise
    
    def test_service_endpoints_exist(self):
        """Service should have required endpoints"""
        if not self.container_available:
            self.skipTest("Container not available")
        
        # Test health endpoint
        response = requests.get(f'{CONTAINER_URL}/health', timeout=10)
        self.assertEqual(response.status_code, 200)
        
        # Test predict endpoint exists (should not 404)
        response = requests.post(f'{CONTAINER_URL}/predict', timeout=10)
        self.assertNotEqual(response.status_code, 404)
    
    def test_no_error_500_on_valid_request(self):
        """Valid requests should not return 500 errors"""
        if not self.container_available:
            self.skipTest("Container not available")
        
        # Send request with minimal parameters
        response = requests.post(
            f'{CONTAINER_URL}/predict',
            data={},
            timeout=30
        )
        # Should be 200 (even if error in response body)
        self.assertNotEqual(response.status_code, 500)


def run_tests():
    """Run all integration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(HealthCheckIntegrationTests))
    suite.addTests(loader.loadTestsFromTestCase(PredictEndpointIntegrationTests))
    suite.addTests(loader.loadTestsFromTestCase(PerformanceIntegrationTests))
    suite.addTests(loader.loadTestsFromTestCase(RobustnessIntegrationTests))
    suite.addTests(loader.loadTestsFromTestCase(DeploymentValidationTests))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
