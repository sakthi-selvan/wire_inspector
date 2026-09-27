import unittest
import numpy as np
from PIL import Image
from engine import summarize, png_bytes, read_image

class Masks(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (20, 20), 'white')
        self.maps = np.zeros((6, 20, 20), dtype=np.float32)
    def test_no_wire_is_not_normal(self):
        self.maps[0] = 0.9
        status, rows, _, mask = summarize(self.image, self.maps)
        self.assertIn('Unable', status)
        self.assertEqual(rows, [])
        self.assertEqual(np.array(mask).max(), 0)
    def test_overlap_and_export(self):
        self.maps[4] = 0.9
        self.maps[0, 2:12, 2:12] = 0.8
        self.maps[1, 2:12, 2:12] = 0.9
        status, rows, _, mask = summarize(self.image, self.maps)
        self.assertEqual([r['defect'] for r in rows], ['Cut'])
        self.assertEqual(np.array(mask)[5, 5], 2)
        self.assertEqual(read_image(png_bytes(mask)).size, (20, 20))
    def test_uncertain_and_normal_candidate(self):
        self.maps[4] = 0.9
        self.assertIn('Uncertain', summarize(self.image, self.maps)[0])
        self.maps[5] = 0.9
        self.assertIn('Normal candidate', summarize(self.image, self.maps)[0])
    def test_small_noise_removed(self):
        self.maps[4] = 0.9
        self.maps[0, 0, 0] = 0.99
        self.assertEqual(summarize(self.image, self.maps)[1], [])

if __name__ == '__main__':
    unittest.main()
