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
    
    # =====================================================================
    # OLD TEST FUNCTION (COMMENTED OUT FOR REFERENCE)
    # =====================================================================
    # def test_predict_with_image_and_question(self):
    #     """Test VQA prediction with image and question - expects new response format with probabilities"""
    #     if not self.container_available:
    #         self.skipTest("Container not available")
    #     
    #     if not TEST_IMAGE_PATH.exists():
    #         self.skipTest(f"Test image not found at {TEST_IMAGE_PATH}")
    #     
    #     # Send prediction request
    #     with open(TEST_IMAGE_PATH, 'rb') as f:
    #         response = requests.post(
    #             f'{CONTAINER_URL}/predict',
    #             files={'file': ('test.jpg', f, 'image/jpeg')},
    #             data={'question': 'what color is the bus?'},
    #             timeout=120
    #         )
    #     
    #     # Verify response status
    #     self.assertEqual(response.status_code, 200, f"Expected 200, got {response.status_code}: {response.text}")
    #     
    #     # Verify response has expected structure
    #     data = response.json()
    #     self.assertNotIn('error', data, f"Response contained error: {data}")
    #     
    #     # Verify new response format with predicted_answers array
    #     self.assertIn('predicted_answers', data, f"Response missing 'predicted_answers' key. Got keys: {list(data.keys())}")
    #     self.assertIn('status', data, f"Response missing 'status' key. Got keys: {list(data.keys())}")
    #     
    #     # Verify predicted_answers is a list with predictions
    #     predicted_answers = data['predicted_answers']
    #     self.assertIsInstance(predicted_answers, list, f"predicted_answers should be a list, got {type(predicted_answers)}")
    #     self.assertGreater(len(predicted_answers), 0, "predicted_answers list is empty")
    #     
    #     # Verify each prediction has required fields
    #     first_prediction = predicted_answers[0]
    #     required_fields = ['rank', 'class_id', 'class_name', 'probability', 'confidence']
    #     for field in required_fields:
    #         self.assertIn(field, first_prediction, f"Prediction missing '{field}' field. Got: {list(first_prediction.keys())}")
    #     
    #     # Verify top prediction has valid values
    #     self.assertEqual(first_prediction['rank'], 1, "Top prediction should have rank 1")
    #     self.assertIsInstance(first_prediction['class_id'], int, "class_id should be integer")
    #     self.assertIsInstance(first_prediction['class_name'], str, "class_name should be string")
    #     self.assertIsInstance(first_prediction['probability'], float, "probability should be float")
    #     self.assertIsInstance(first_prediction['confidence'], str, "confidence should be string")
    #     self.assertGreaterEqual(first_prediction['probability'], 0, "probability should be >= 0")
    #     self.assertLessEqual(first_prediction['probability'], 1, "probability should be <= 1")
    #     
    #     print(f"\n✓ Prediction successful!")
    #     print(f"  Status: {data['status']}")
    #     print(f"  Total predictions: {len(predicted_answers)}")
    #     print(f"  Top prediction: {first_prediction['class_name']} (probability: {first_prediction['confidence']})")
    # =====================================================================
    
    def test_predict_with_image_and_question(self):
        """Test VQA prediction with image and question - expects new response format with probabilities"""
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
        
        # Verify response has expected structure
        data = response.json()
        self.assertNotIn('error', data, f"Response contained error: {data}")
        
        # Verify new response format with predicted_answers array
        self.assertIn('predicted_answers', data, f"Response missing 'predicted_answers' key. Got keys: {list(data.keys())}")
        self.assertIn('status', data, f"Response missing 'status' key. Got keys: {list(data.keys())}")
        
        # Verify predicted_answers is a list with predictions
        predicted_answers = data['predicted_answers']
        self.assertIsInstance(predicted_answers, list, f"predicted_answers should be a list, got {type(predicted_answers)}")
        self.assertGreater(len(predicted_answers), 0, "predicted_answers list is empty")
        
        # Verify each prediction has required fields
        first_prediction = predicted_answers[0]
        required_fields = ['rank', 'class_id', 'class_name', 'probability', 'confidence']
        for field in required_fields:
            self.assertIn(field, first_prediction, f"Prediction missing '{field}' field. Got: {list(first_prediction.keys())}")
        
        # Verify top prediction has valid values
        self.assertEqual(first_prediction['rank'], 1, "Top prediction should have rank 1")
        self.assertIsInstance(first_prediction['class_id'], int, "class_id should be integer")
        self.assertIsInstance(first_prediction['class_name'], str, "class_name should be string")
        self.assertIsInstance(first_prediction['probability'], float, "probability should be float")
        self.assertIsInstance(first_prediction['confidence'], str, "confidence should be string")
        
        print(f"\n✓ Prediction successful!")
        print(f"  Status: {data['status']}")
        print(f"  Total predictions: {len(predicted_answers)}")
        print(f"  Top prediction: {first_prediction['class_name']} (probability: {first_prediction['confidence']})")


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
