# -*- coding: utf-8 -*-
"""Chapter 2"""

import ssl
import os
import tarfile
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from six.moves import urllib
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.tools.plotting import scatter_matrix

ssl._create_default_https_context = ssl._create_unverified_context

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Download data for the project"""
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    """Load the housing data into pandas"""
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def split_train_test(data, test_ratio, seed=42):
    """Splits the data into train and test using the test_ratio"""
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio, hash):
    """Check if the current identifier is in the test or the train set"""
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    """Split using the hashing and some identifier"""
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

fetch_housing_data()

housing = load_housing_data()

print(housing.head(), '\n')
print(housing.columns, '\n')
print(housing.info(), '\n')
print(housing["ocean_proximity"].value_counts(), '\n')
print(housing.describe(), '\n')

## dataframe.hist() historgram of all numericals
## check how to select matplotlib backend!
# housing.hist(bins=50, figsize=(20, 15))

## Regular split of test and train dataset
train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")

## Split using unique identifiers
housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

## Split using unique identifiers
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

## Using scikit train/test splitter
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

## discretizing median income
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

## Splitting the data by strata
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

## Checking if the proportions are correct in the three groups
print(housing["income_cat"].value_counts()/len(housing))
print(strat_train_set["income_cat"].value_counts()/len(strat_train_set))
print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))

## Removing the strata category from the data
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"],axis=1, inplace=True)

## with the sets created, replace original hosing with train set
housing = strat_train_set.copy()

## plotting directly with pandas
housing.plot(kind="scatter",
             x="longitude",
             y="latitude",
             alpha=0.4,
             s=housing["population"]/100,
             label="population",
             c="median_house_value",
             cmap=plt.get_cmap("jet"),
             colorbar=True
             )
plt.legend()

## getting correlation using only pandas
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

## plotting the SPLOM with pandas
attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(housing[attributes],figsize=(12,8))

# scatterplot using pandas again
housing.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)

# adding some attribute combinations
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

# check the new correlation_matrix
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
attributes = ["median_house_value",
              "median_income",
              "housing_median_age",
              "population_per_household",
              "bedrooms_per_room",
              "rooms_per_household"]
scatter_matrix(housing[attributes],figsize=(12,8))

## Restoring data to initial state
housing = strat_train_set.drop("median_house_value", axis = 1)
housing_labels = strat_train_set["median_house_value"].copy() # copy not point

## Data Cleaning using Scikit tools