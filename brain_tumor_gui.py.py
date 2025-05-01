import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import numpy as np

# Assuming your file is named 'brain_tumor_gui.py'
import MRI

class TestBrainTumorSegmentation(unittest.TestCase):

    @patch('brain_tumor_gui.filedialog.askdirectory')
    def test_upload(self, mock_askdirectory):
        mock_askdirectory.return_value = 'test_dataset'
        brain_tumor_gui.upload()
        self.assertEqual(brain_tumor_gui.filename, 'test_dataset')

    @patch('os.path.exists')
    @patch('numpy.load')
    def test_generate_model_load_existing(self, mock_np_load, mock_path_exists):
        mock_path_exists.return_value = True
        mock_np_load.return_value = np.zeros((10, 128, 128, 1))
        brain_tumor_gui.generateModel()
        self.assertEqual(len(brain_tumor_gui.X), 10)
        self.assertEqual(len(brain_tumor_gui.Y), 10)

    @patch('os.path.exists')
    @patch('os.walk')
    @patch('cv2.imread')
    def test_generate_model_from_folder(self, mock_imread, mock_walk, mock_path_exists):
        mock_path_exists.return_value = False
        mock_walk.return_value = [('.', [], ['img1.jpg', 'img2.jpg'])]
        mock_imread.return_value = np.zeros((128,128), dtype=np.uint8)

        brain_tumor_gui.filename = 'dummy_folder'
        brain_tumor_gui.generateModel()
        self.assertGreater(len(brain_tumor_gui.X), 0)
        self.assertGreater(len(brain_tumor_gui.Y), 0)

    @patch('os.path.exists')
    @patch('brain_tumor_gui.model_from_json')
    @patch('builtins.open', new_callable=mock_open, read_data='{}')
    @patch('pickle.load')
    def test_cnn_load_existing_model(self, mock_pickle_load, mock_open_func, mock_model_from_json, mock_path_exists):
        mock_path_exists.return_value = True
        mock_pickle_load.return_value = {'accuracy': [0.8]*10}

        model_mock = MagicMock()
        model_mock.summary = MagicMock()
        mock_model_from_json.return_value = model_mock

        brain_tumor_gui.X = np.zeros((100, 128, 128, 1))
        brain_tumor_gui.Y = np.zeros(100)
        brain_tumor_gui.CNN()
        self.assertTrue(brain_tumor_gui.accuracy > 0)

    @patch('brain_tumor_gui.ftplib.FTP_TLS')
    def test_get_images(self, mock_ftp_tls):
        ftp_mock = MagicMock()
        ftp_mock.nlst.return_value = ['image1.jpg', 'image2.jpg']
        mock_ftp_tls.return_value = ftp_mock

        brain_tumor_gui.getImages()
        self.assertIn('image1.jpg', brain_tumor_gui.value)
        self.assertIn('image2.jpg', brain_tumor_gui.value)

    @patch('brain_tumor_gui.ftplib.FTP_TLS')
    @patch('brain_tumor_gui.classifier')
    @patch('cv2.imread')
    @patch('cv2.resize')
    @patch('cv2.putText')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    def test_predict(self, mock_waitkey, mock_imshow, mock_puttext, mock_resize, mock_imread, mock_classifier, mock_ftp_tls):
        ftp_mock = MagicMock()
        mock_ftp_tls.return_value = ftp_mock
        brain_tumor_gui.imagelist = MagicMock()
        brain_tumor_gui.imagelist.get.return_value = 'some_image.jpg'

        mock_classifier.predict.return_value = np.array([[0.1, 0.9]])
        mock_imread.return_value = np.zeros((128, 128), dtype=np.uint8)
        mock_resize.return_value = np.zeros((800, 500, 3), dtype=np.uint8)

        brain_tumor_gui.predict()
        self.assertTrue(mock_classifier.predict.called)

if __name__ == '__main__':
    unittest.main()
