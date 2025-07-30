"""
Unit tests for diabetes prediction utility functions.

This module contains tests to ensure the reliability and correctness
of the utility functions used in the diabetes prediction project.
"""

import unittest
import pandas as pd
import numpy as np
from utils import parse_categorical_data, create_age_groups, preprocess_diabetes_data


class TestDiabetesUtils(unittest.TestCase):
    """Test cases for diabetes prediction utility functions."""

    def setUp(self):
        """Set up test data for unit tests."""
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'Age': [25, 45, 60, 35],
            'Gender': ['Male', 'Female', 'Male', 'Female'],
            'Polyuria': ['Yes', 'No', 'Yes', 'No'],
            'class': ['Positive', 'Negative', 'Positive', 'Negative']
        })

    def test_parse_categorical_data_positive_cases(self):
        """Test parse_categorical_data with positive indicators."""
        self.assertEqual(parse_categorical_data('Yes'), 1)
        self.assertEqual(parse_categorical_data('Positive'), 1)
        self.assertEqual(parse_categorical_data('Female'), 1)

    def test_parse_categorical_data_negative_cases(self):
        """Test parse_categorical_data with negative indicators."""
        self.assertEqual(parse_categorical_data('No'), 0)
        self.assertEqual(parse_categorical_data('Negative'), 0)
        self.assertEqual(parse_categorical_data('Male'), 0)

    def test_parse_categorical_data_unchanged_cases(self):
        """Test parse_categorical_data with values that should remain unchanged."""
        self.assertEqual(parse_categorical_data(25), 25)
        self.assertEqual(parse_categorical_data('Unknown'), 'Unknown')

    def test_create_age_groups(self):
        """Test age group creation function."""
        self.assertEqual(create_age_groups(18), 'G6-20')
        self.assertEqual(create_age_groups(25), 'G21-35')
        self.assertEqual(create_age_groups(40), 'G36-50')
        self.assertEqual(create_age_groups(55), 'G51-65')

    def test_preprocess_diabetes_data_shape(self):
        """Test that preprocessing maintains data shape."""
        processed_data = preprocess_diabetes_data(self.sample_data)
        self.assertEqual(processed_data.shape, self.sample_data.shape)

    def test_preprocess_diabetes_data_conversion(self):
        """Test that categorical data is properly converted to numerical."""
        processed_data = preprocess_diabetes_data(self.sample_data)
        
        # Check that 'Male' becomes 0 and 'Female' becomes 1
        expected_gender = [0, 1, 0, 1]
        self.assertEqual(list(processed_data['Gender']), expected_gender)
        
        # Check that 'Yes' becomes 1 and 'No' becomes 0
        expected_polyuria = [1, 0, 1, 0]
        self.assertEqual(list(processed_data['Polyuria']), expected_polyuria)

    def test_preprocess_removes_duplicates(self):
        """Test that preprocessing removes duplicate rows."""
        # Create data with duplicates
        duplicate_data = pd.DataFrame({
            'Age': [25, 25, 45],
            'Gender': ['Male', 'Male', 'Female'],
            'class': ['Positive', 'Positive', 'Negative']
        })
        
        processed_data = preprocess_diabetes_data(duplicate_data)
        
        # Should have only 2 unique rows after removing duplicates
        self.assertEqual(len(processed_data), 2)

    def test_age_boundary_conditions(self):
        """Test age group function with boundary conditions."""
        self.assertEqual(create_age_groups(20), 'G6-20')
        self.assertEqual(create_age_groups(21), 'G21-35')
        self.assertEqual(create_age_groups(35), 'G21-35')
        self.assertEqual(create_age_groups(36), 'G36-50')


class TestDataValidation(unittest.TestCase):
    """Test cases for data validation functions."""

    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes."""
        empty_df = pd.DataFrame()
        processed = preprocess_diabetes_data(empty_df)
        self.assertTrue(processed.empty)

    def test_single_row_dataframe(self):
        """Test processing of single-row dataframe."""
        single_row = pd.DataFrame({
            'Age': [30],
            'Gender': ['Female'],
            'class': ['Positive']
        })
        processed = preprocess_diabetes_data(single_row)
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed.iloc[0]['Gender'], 1)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
