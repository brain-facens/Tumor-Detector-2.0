# libraries
from src.net import TumorSegmentationDetector, tf

import unittest



class TestProject(unittest.TestCase):
    
    def test_TumorSegmentationDetector(self):
        # check if the output shape match to the expected shape passed below
        self.assertEqual(
            TumorSegmentationDetector()(tf.zeros(shape=[1, 256, 256, 3], dtype=tf.float32)).shape,
            tf.zeros(shape=[1, 256, 256, 1], dtype=tf.int32).shape
        )



if __name__ == '__main__':
    unittest.main()