import shelve
import pandas as pd

DEFAULT_DB = '/home/louis/Data/kaggle-africa-soil/default.db'

class AbstractPreprocessor(object):
    def __init__(self, db_name=None):
        if not db_name:
            db_name=DEFAULT_DB
        self.cache = shelve.open(db_name)
        self.name = self.__class__.__name__

    def _transform(self, data):
        raise NotImplementedError

    def run(self, data):
        cached_data = self._get_cached_transform(data)
        if cached_data is not None:
            return cached_data
        transformed_data = self._transform(data)
        self._cache_transform(transformed_data)
        return transformed_data

    def _cache_transform(self, data):
        pass

    def _get_cached_transform(self, data):
        return None

    def _cache_value(self, key, value):
        self.cache[str(key)] = value

    def _get_cached_value(self, key):
        str_key = str(key)
        if str_key in self.cache:
            return self.cache[str_key]
        return None

class AbstractFrameCachePreprocessor(AbstractPreprocessor):
    """A cache class which performs caching for an entire set of IDs. Assumes that each row has an ID column.

    This should be used when the full set of IDs is needed to properly retrieve the data,
    rather than just row information. Inheritors of this class must populate the cache_columns field to use caching."""
    def __init__(self, db_name=None):
        super(AbstractFrameCachePreprocessor, self).__init__(db_name=db_name)
        self.cache_columns = None

    def _cache_transform(self, data):
        sorted_ids = tuple(sorted(data.index))
        # Get subset, flatten and push to disk
        flattened_data = self._extract_flattened_data(data)
        self._cache_value(sorted_ids, flattened_data)

    def _extract_flattened_data(self, data):
        data_dictionary = data.to_dict()
        flattened_data = [(c,data_dictionary[c]) for c in self.cache_columns]
        return flattened_data

    def _get_cached_transform(self, data):
        sorted_ids = tuple(sorted(data.index))
        # Retrieve from disk, unflatten, and combine with data
        flattened_data = self._get_cached_value(sorted_ids)
        if not flattened_data:
            return None
        cached_dataframe = self._flattened_data_to_dataframe(flattened_data)
        return data.combineAdd(cached_dataframe)

    def _flattened_data_to_dataframe(self, flattened_data):
        return pd.DataFrame(dict(flattened_data))
