import unittest
import numpy as np
import pandas as pd
from src.preprocessing import check_normality, apply_transformations


class TestPreprocessingPipeline(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Create dummy skewed data containing zeros to replicate the true dataset environment
        self.df = pd.DataFrame({
            "FeatureA": np.random.exponential(scale=2.0, size=100),
            "FeatureB": np.append(np.random.exponential(scale=5.0, size=90), np.zeros(10)),
            "Outcome": np.random.choice([0, 1], size=100)
        })

    def test_normality_check_returns_dictionary(self):
        report = check_normality(self.df, "FeatureA")
        self.assertIn("skewness", report)
        self.assertIn("is_normal", report)

    def test_adaptive_transformations_reduce_skew(self):
        features = ["FeatureA", "FeatureB"]
        transformed_df, summary = apply_transformations(self.df, features)
        
        # Verify columns with zero markers trigger Yeo-Johnson instead of failing on Box-Cox
        self.assertEqual(summary["FeatureB"]["method"], "Yeo-Johnson")
        
        # Check that skewness was successfully minimized closer to zero
        self.assertLess(abs(summary["FeatureA"]["transformed_skew"]), abs(summary["FeatureA"]["original_skew"]))


if __name__ == "__main__":
    unittest.main()
