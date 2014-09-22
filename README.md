dataframe-preprocessor
======================

A tool for preprocessing pandas dataframes and caching preprocessed data.

Includes two abstract base classes, which can be extended to perform arbitrary caching and preprocessing. One of the base classes persists preprocessed columns of the dataframe using the spectacular (shelve)[https://docs.python.org/2/library/shelve.html] module from the Python Standard Library.