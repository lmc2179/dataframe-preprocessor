import pickledb

DEFAULT_DB = '/home/louis/Data/kaggle-africa-soil/default.db'

class AbstractPreprocessor(object):
    def __init__(self, db_name=None):
        if not db_name:
            db_name=DEFAULT_DB
        self.cache = pickledb.load(db_name, False)
        self.name = self.__class__.__name__

    def _transform(self, data):
        raise NotImplementedError

    def run(self, data):
        cached_data = self._get_cached_transform(data)
        if cached_data:
            return cached_data
        transformed_data = self._transform(data)
        self._cache_transform(transformed_data)
        return transformed_data

    def _cache_transform(self, data):
        pass

    def _get_cached_transform(self, data):
        return None

    def _cache_value(self, key, value):
        self.cache.set(key, value)

    def _get_cached_value(self, key):
        return self.cache.get(key)

class AbstractRowCachePreprocessor(AbstractPreprocessor):
    "A cache class which performs caching by row. Assumes that each row has an ID column."
    pass


