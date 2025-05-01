import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import cv2
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
import tkinter as tk
from MRI import *  # Importing the source code file

class TestMRI(unittest.TestCase):

    @patch('tkinter.filedialog.askdirectory')  # Mocking file dialog
    def test_upload(self, mock_askdirectory):
        mock_askdirectory.return_value = "mock_directory"
        upload()
        self.assertEqual(filename, "mock_directory")
        self.assertIn("mock_directory loaded", text.get("1.0", "end-1c"))

    @patch('os.path.exists')  # Mocking file existence check
    @patch('numpy.load')  # Mocking numpy load function
    def test_generateModel(self, mock_load, mock_exists):
        mock_exists.return_value = True  # Simulating the existence of the file
        mock_load.return_value = np.array([1, 2, 3])  # Mocking numpy load return value
        
        generateModel()
        
        # Verifying the outputs after function call
        self.assertTrue(len(X) > 0)
        self.assertTrue(len(Y) > 0)
        self.assertIn("Total number of images found in dataset", text.get("1.0", "end-1c"))
        self.assertIn("Total number of classes", text.get("1.0", "end-1c"))

    @patch('keras.models.model_from_json')  # Mocking model loading
    @patch('numpy.load')  # Mocking numpy load
    @patch('keras.models.Sequential')  # Mocking the Keras model
    def test_CNN(self, mock_sequential, mock_load, mock_model_from_json):
        mock_load.return_value = np.array([1, 2, 3])  # Mocking numpy load
        mock_sequential.return_value = MagicMock()  # Mocking the model creation
        
        # Simulating an already trained model scenario
        mock_model_from_json.return_value = MagicMock()
        mock_model_from_json.return_value.load_weights.return_value = None
        
        CNN()
        
        # Check for accuracy text inserted
        self.assertIn("CNN Prediction Accuracy", text.get("1.0", "end-1c"))

    @patch('cv2.imread')  # Mocking the imread function
    @patch('keras.models.Sequential.predict')  # Mocking the predict method of Keras model
    @patch('ftplib.FTP_TLS')  # Mocking FTP functionality
    def test_predict(self, mock_FTP, mock_predict, mock_imread):
        # Mock FTP functionality
        mock_ftp_instance = MagicMock()
        mock_FTP.return_value = mock_ftp_instance
        mock_ftp_instance.nlst.return_value = ['mock_image.jpg']
        
        # Mock imread function
        mock_imread.return_value = np.zeros((128, 128), dtype=np.uint8)  # Returning a black image
        
        # Mocking the prediction to return a value indicating a tumor
        mock_predict.return_value = np.array([[0, 1]])
        
        predict()
        
        # Verifying the text in the output after prediction
        self.assertIn("Disease Identified as : Tumor Detected", text.get("1.0", "end-1c"))

    @patch('ftplib.FTP_TLS')  # Mocking FTP functionality
    def test_getImages(self, mock_FTP):
        mock_ftp_instance = MagicMock()
        mock_FTP.return_value = mock_ftp_instance
        mock_ftp_instance.nlst.return_value = ['image1.jpg', 'image2.jpg']
        
        getImages()
        
        # Verifying the updated value in the combobox after fetching images
        self.assertIn('image1.jpg', imagelist['values'])
        self.assertIn('image2.jpg', imagelist['values'])

    def test_initial_conditions(self):
        # Test initial setup of variables, ensuring they are empty
        self.assertEqual(len(X), 0)
        self.assertEqual(len(Y), 0)
        self.assertIsNone(filename)
        self.assertIsNone(classifier)

    def test_disease(self):
        # Testing if disease list is set properly
        self.assertEqual(disease[0], 'No Tumor Detected')
        self.assertEqual(disease[1], 'Tumor Detected')

if __name__ == '__main__':
    unittest.main()
