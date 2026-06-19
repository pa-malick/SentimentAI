"""
Tests unitaires pour les fonctions de preprocessing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from src.preprocessing import clean_text, preprocess_batch


class TestCleanText(unittest.TestCase):

    def test_lowercase(self):
        result = clean_text("Hello World")
        self.assertEqual(result, result.lower())

    def test_remove_html(self):
        result = clean_text("<br>Great movie!</br>")
        self.assertNotIn("<br>", result)
        self.assertNotIn("</br>", result)

    def test_remove_punctuation(self):
        result = clean_text("This is great!!!")
        self.assertNotIn("!", result)

    def test_remove_stopwords(self):
        result = clean_text("this is a very good film")
        self.assertNotIn("this", result)
        self.assertNotIn("is", result)
        self.assertNotIn("a", result)

    def test_short_words_removed(self):
        result = clean_text("I am ok at")
        tokens = result.split()
        for token in tokens:
            self.assertGreater(len(token), 2)

    def test_noisy_text(self):
        result = clean_text("WOW!!! Amazing 123 film <p>Best</p> ever!!!")
        self.assertGreater(len(result), 0)
        self.assertNotIn("!!!", result)

    def test_empty_string(self):
        result = clean_text("")
        self.assertEqual(result, "")

    def test_preprocess_batch(self):
        texts = ["Great film", "Terrible movie"]
        results = preprocess_batch(texts, verbose=False)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], str)


class TestEdgeCases(unittest.TestCase):

    def test_only_html(self):
        result = clean_text("<div><p><br></p></div>")
        self.assertEqual(result.strip(), "")

    def test_numbers_removed(self):
        result = clean_text("There are 100 reasons to watch this")
        self.assertNotIn("100", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
