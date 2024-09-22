import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from gboost_module.models import forecasting

# Import your CustomEstimator class
# from your_module import CustomEstimator  # Replace 'your_module' with the actual module name

# For the purpose of this test, I'll assume CustomEstimator is defined in the same script.
# If it's in a different module, you should import it accordingly.


class TestForecast(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data
        num_samples = 200
        self.dates = [datetime.now() + timedelta(days=i) for i in range(num_samples)]
        self.y = np.sin(np.linspace(0, 20, num_samples))  # Synthetic target variable
        self.df = pd.DataFrame({"ds": self.dates, "y": self.y})

    def test_fit_predict(self):
        """
        Test that the estimator can fit and predict without errors.
        """
        # Split data into training and testing sets
        train_df = self.df.iloc[:150]
        test_df = self.df.iloc[150:]

        # Initialize and fit the estimator
        estimator = forecasting.Forecast(nrounds=100)
        estimator.fit(train_df[["ds"]], train_df["y"])

        # Make predictions
        predictions = estimator.predict(test_df[["ds"]])

        # Check that predictions have the correct length
        self.assertEqual(len(predictions), len(test_df), "Prediction length mismatch.")

        # Check that predictions are not NaN
        self.assertFalse(np.isnan(predictions).any(), "Predictions contain NaN values.")

        # Optionally, check that predictions are numerical
        self.assertTrue(
            np.issubdtype(predictions.dtype, np.number),
            "Predictions are not numerical.",
        )

    def test_loss_decrease(self):
        """
        Test that the training loss decreases over epochs.
        """
        # Use only the training data
        train_df = self.df.iloc[:150]

        # Initialize and fit the estimator
        estimator = forecasting.Forecast(nrounds=100)
        estimator.fit(train_df[["ds"]], train_df["y"])

        # Get the recorded losses
        losses = estimator.losses_

        # Check that losses have been recorded
        self.assertTrue(len(losses) > 0, "No losses recorded during training.")

        # Check that the loss decreases over time
        initial_loss = losses[0]
        final_loss = losses[-1]
        self.assertLess(
            final_loss, initial_loss, "Final loss is not less than initial loss."
        )

        # Check that the losses are decreasing monotonically (not required but useful)
        self.assertTrue(
            all(x >= y for x, y in zip(losses[:10], losses[1:11])),
            "Losses are not monotonically decreasing.",
        )


class TestRecursiveSplit:
    def test_case_1(self):
        # Test input
        df = pd.DataFrame({"value": [10, 20, 30, 40]})
        result = forecasting.recursive_split(df, "value", depth=2)

        # Expected output
        expected = pd.DataFrame(
            {
                "value": [10, 20, 30, 40],
                "split_1": [0, 0, 1, 1],
                "split_2": [0, 1, 0, 1],
            }
        )

        # Assert equality
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_case_2(self):
        # Test input
        df = pd.DataFrame({"value": [5, 15, 25, 35, 45, 55, 65, 75]})
        result = forecasting.recursive_split(df, "value", depth=3)

        # Expected output
        expected = pd.DataFrame(
            {
                "value": [5, 15, 25, 35, 45, 55, 65, 75],
                "split_1": [0, 0, 0, 0, 1, 1, 1, 1],
                "split_2": [0, 0, 1, 1, 0, 0, 1, 1],
                "split_3": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        # Assert equality
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
