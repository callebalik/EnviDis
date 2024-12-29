import pandas as pd
import os
import unittest
import sys

# Add the project directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add scripts directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from scripts.csv_to_cooccurence_matrix import create_cooccurrence_matrix

class TestCsvToCooccurrenceMatrix(unittest.TestCase):

    def setUp(self):
        self.input_filepath = 'tests/data/temp/cooc50.csv'
        self.test_output_filepath = 'tests/data/generated/test_cooc50_matrix.csv'
        self.limit_rows = 100
        self.limit_columns = 100

        # Create a sample input CSV file
        sample_data = {
            'entity_1': ['cancer', 'fire', 'depression', 'anxiety', 'pain', 'tumor', 'asthma', 'diabetes'],
            'entity_2': ['current', 'fire', 'current', 'current', 'current', 'current', 'current', 'current'],
            'fq': [50112, 41171, 32291, 23458, 22517, 20810, 17442, 16314]
        }

        unique_entity_1 = list(set(sample_data['entity_1']))
        unique_entity_2 = list(set(sample_data['entity_2']))

        self.row_dimension = len(unique_entity_2)
        self.column_dimension = len(unique_entity_1)

        df = pd.DataFrame(sample_data)
        df.to_csv(self.input_filepath, index=False)

        # Remove the test output file if it exists
        if os.path.exists(self.test_output_filepath):
            os.remove(self.test_output_filepath)

    def tearDown(self):
        pass

    def test_create_cooccurrence_matrix(self):
        # Run the function to create the co-occurrence matrix
        create_cooccurrence_matrix(self.input_filepath, self.test_output_filepath, self.limit_rows, self.limit_columns)

        # Read the output CSV file
        result_df = pd.read_csv(self.test_output_filepath, header=0, index_col=0)

        # Check if the matrix has the correct shape
        self.assertEqual(result_df.shape, (self.row_dimension, self.column_dimension))

        # Check if the values are correctly aggregated
        self.assertEqual(result_df.loc['current', 'cancer'], 50112)
        self.assertEqual(result_df.loc['fire', 'fire'], 41171)
        self.assertEqual(result_df.loc['current', 'depression'], 32291)
        self.assertEqual(result_df.loc['current', 'anxiety'], 23458)
        self.assertEqual(result_df.loc['current', 'pain'], 22517)
        self.assertEqual(result_df.loc['current', 'tumor'], 20810)
        self.assertEqual(result_df.loc['current', 'asthma'], 17442)
        self.assertEqual(result_df.loc['current', 'diabetes'], 16314)

if __name__ == '__main__':
    unittest.main()
