import preprocess
import unittest
import pandas as pd
from copy import deepcopy

class PolynomialPreprocessor(preprocess.AbstractPreprocessor):
    "Test class which augments a dataframe with polynomial features."
    def __init__(self, db_name=None, degree=None, feature=None):
        super(PolynomialPreprocessor, self).__init__(db_name)
        self.degree = degree
        self.feature = feature

    def _transform(self, data):
        data = deepcopy(data)
        data['unit'] = 1
        for i in range(2,self.degree+1):
            column_name = 'poly_{0}'.format(i)
            data[column_name] = data[self.feature]**i
        return data

class AvgDistancePreprocessor(preprocess.AbstractFrameCachePreprocessor):
    def __init__(self, db_name=None, target_column=None):
        super(AvgDistancePreprocessor, self).__init__(db_name)
        self.cache_columns = ['dist_from_avg']
        self.target_column = target_column

    def _transform(self, data):
        data = deepcopy(data)
        avg = data[self.target_column].mean()
        data['dist_from_avg'] = abs(data[self.target_column] - avg)
        return data

class PolynomialTest(unittest.TestCase):
    def test_transform(self):
        test_data = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6]}, index=[0,1,2])
        expected_output = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6], 'unit':[1,1,1], 'poly_2':[1, 4, 9]}, index=[0,1,2])
        preproc = PolynomialPreprocessor(degree=2, feature='a')
        output = preproc.run(test_data)
        assert (expected_output.sort(axis=1) == output.sort(axis=1)).all().all()

class AvgTest(unittest.TestCase):
    def test_transform(self):
        test_data = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6]}, index=[0,1,2])
        expected_output = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6], 'dist_from_avg':[1,0,1]}, index=[0,1,2])
        preproc = AvgDistancePreprocessor(target_column='a')
        output = preproc.run(test_data)
        assert (expected_output.sort(axis=1) == output.sort(axis=1)).all().all()

if __name__=='__main__':
    unittest.main()