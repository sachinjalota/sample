import unittest
from src.models.text_input import TextCompletionRequest



class MyTestCase(unittest.TestCase):

    def test_text_completion(self):
        request = TextCompletionRequest()
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
