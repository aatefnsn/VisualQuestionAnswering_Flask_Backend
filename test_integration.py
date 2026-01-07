"""
Integration tests for VQA backend - tests deployed container
Run this test against a live deployment with: CONTAINER_URL=https://your-url pytest test_integration.py -v
"""
import os
import sys
import unittest
import requests
import time
from pathlib import Path

# Get container URL from environment
CONTAINER_URL = os.getenv('CONTAINER_URL', 'http://localhost:8080')

# Get test image path
TEST_IMAGE_DIR = Path(__file__).parent / 'test'
TEST_IMAGE_PATH = TEST_IMAGE_DIR / 'COCO_train2014_000000000081.jpg'


class VQAPredictionTest(unittest.TestCase):
    """Simple test for VQA prediction with image and question"""
    
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
        raise Exception(f"Container not available at {CONTAINER_URL}")
    
    def test_predict_with_image_and_question(self):
        """Test VQA prediction with image and question"""
        if not self.container_available:
            self.skipTest("Container not available")
        
        if not TEST_IMAGE_PATH.exists():
            self.skipTest(f"Test image not found at {TEST_IMAGE_PATH}")
        
        # Send prediction request
        with open(TEST_IMAGE_PATH, 'rb') as f:
            response = requests.post(
                f'{CONTAINER_URL}/predict',
                files={'file': ('test.jpg', f, 'image/jpeg')},
                data={'question': 'what color is the bus?'},
                timeout=120
            )
        
        # Verify response status
        self.assertEqual(response.status_code, 200, f"Expected 200, got {response.status_code}: {response.text}")
        
        # Verify response has predicted answers
        data = response.json()
        self.assertNotIn('error', data, f"Response contained error: {data}")
        
        # Verify response contains class predictions (class_name-0, class_name-1, etc.)
        has_predictions = any(key.startswith('class_name-') for key in data.keys())
        self.assertTrue(has_predictions, f"Response missing class predictions. Got keys: {list(data.keys())}")
        
        print(f"\n✓ Prediction successful!")
        print(f"  Predicted answers: {list(data.keys())[:3]}...")


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(VQAPredictionTest))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
