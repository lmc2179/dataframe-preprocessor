import preprocess
from copy import deepcopy
from sklearn import decomposition

class PCAPreprocessor(preprocess.AbstractFrameCachePreprocessor):
    def __init__(self, db_name=None, target_columns=None, number_components=None):
        super(PCAPreprocessor, self).__init__(db_name=db_name)
        self.target_columns = set(target_columns)
        self.number_components = number_components
        self.cache_columns = ['pca_comp_{0}'.format(i+1) for i in range(number_components)]

    def _transform(self, data):
        pca = decomposition.PCA(n_components=self.number_components)
        pca_data = data.drop([c for c in data.columns if c not in self.target_columns], axis=1)
        pca_transformed_data = pca.fit_transform(pca_data.values)
        data = deepcopy(data)
        for i, pca_component_column in enumerate(zip(*pca_transformed_data)):
            data['pca_comp_{0}'.format(i+1)] = pca_component_column
        return data

    def _get_unique_identifier(self, data):
        return str(sorted(data.index)) + '|' + str(self.target_columns)

import pandas as pd

df = pd.DataFrame({'x':[1,1,2,3,4,5,6], 'y':[4,5,6,3,7,3,0], 'z':[2,2,4,6,8,10,12]}, index=[0,1,2,3,4,5,6])
print PCAPreprocessor(target_columns=['x','z'], number_components=2).run(df)
