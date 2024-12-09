import unittest
from unittest.mock import patch, MagicMock
import cv2
import numpy as np
import os
from facerec import train_model, detect_faces, recognize_face

class TestFaceRecognition(unittest.TestCase):

    @patch('facerec.cv2.CascadeClassifier')  # Ensure the patch matches how it's imported in facerec
    def test_detect_faces(self, mock_cascade):
        # Debugging paths and image loading
        print("Directory exists:", os.path.exists('face_samples/akshay kumar'))
        print("1 exists:", os.path.exists('face_samples/akshay kumar/1.png'))
        for image_path in ['face_samples/akshay kumar/1.png', 'face_samples/akshay kumar/2.png']:
            img = cv2.imread(image_path)
            print(f"Loaded {image_path}: {img is not None}")

        # Mocking Haar Cascade detectMultiScale method
        mock_cascade_instance = MagicMock()
        mock_cascade_instance.detectMultiScale.return_value = [(10, 10, 50, 50)]
        mock_cascade.return_value = mock_cascade_instance

        # Create a dummy gray frame for testing
        gray_frame = np.zeros((100, 100), dtype=np.uint8)

        # Debugging the mock before calling the function
        print("Mock detectMultiScale is set to return:", mock_cascade_instance.detectMultiScale.return_value)

        # Call detect_faces function
        faces = detect_faces(gray_frame)

        # Debugging the output of detect_faces
        print("Detected faces:", faces)

        # Verify the output of detect_faces
        self.assertEqual(faces, [(10, 10, 50, 50)])
        mock_cascade_instance.detectMultiScale.assert_called_once()

        # Debugging whether detectMultiScale was called
        print("Was detectMultiScale called?", mock_cascade_instance.detectMultiScale.called)


    @patch('facerec.os.listdir')
    @patch('facerec.os.walk')
    @patch('facerec.cv2.face.LBPHFaceRecognizer.create')
    def test_train_model(self, mock_create, mock_walk, mock_listdir):
        # Mocking os.walk to simulate directory structure
        mock_walk.return_value = [
            ('face_samples', ['akshay kumar'], []),
        ]
        # Mocking os.listdir for the subject path
        mock_listdir.return_value = ['1.png', '2.png']
        
        # Mocking the recognizer's train method
        mock_model = MagicMock()
        mock_create.return_value = mock_model

        # Train the model
        model, names = train_model()

        # Verify names and mock calls
        self.assertEqual(names, {0: 'akshay kumar'})
        mock_model.train.assert_called_once()


    def test_recognize_face(self):
        # Mocking the recognizer
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, 50)  # Simulate a confident match

        # Dummy inputs
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        gray_frame = np.zeros((100, 100), dtype=np.uint8)
        face_coords = [(10, 10, 50, 50)]
        names = {0: 'akshay kumar'}

        # Normalize names to lowercase for consistent comparison
        normalized_names = {k: v.lower() for k, v in names.items()}

        # Call the recognition function
        result_frame, recognized = recognize_face(mock_model, frame, gray_frame, face_coords, normalized_names)

        # Normalize recognized names
        recognized = [(name.lower(), confidence) for name, confidence in recognized]

        # Expected output
        expected_recognized = [('akshay kumar', 50)]

        # Check recognized output and sounds played
        self.assertEqual(recognized, expected_recognized)

if _name_ == '_main_':
    unittest.main()