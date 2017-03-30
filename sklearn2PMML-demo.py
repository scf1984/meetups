# coding: utf-8

import pandas, numpy

from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer, FunctionTransformer
from sklearn.feature_selection import SelectKBest

from sklearn_pandas import DataFrameMapper

from sklearn2pmml import PMMLPipeline, sklearn2pmml


click_data = pandas.read_csv("clicks.csv")[['gender', 'age_18_24', 'country', 'click']]
click_data = click_data.dropna()

click_data.head()

mapper = DataFrameMapper([
    ('gender', None),
    ('age_18_24', None)
])

click_pipeline = Pipeline([
    ('mapper', mapper),
 	('classifier', LogisticRegression())
 ])

click_pipeline = click_pipeline.fit(click_data.drop('click', 1), click_data.click)
click_pipeline.steps[1][1].coef_

mapper = DataFrameMapper([
    ('gender', None),
    ('age_18_24', None)
])

click_pipeline = PMMLPipeline([
    ('mapper', mapper),
 	('classifier', LogisticRegression())
 ])

click_pipeline = click_pipeline.fit(click_data.drop('click', 1), click_data.click)
sklearn2pmml(click_pipeline, "click1.pmml")

mapper = DataFrameMapper([
    ('country', LabelBinarizer()),
    ('gender', None),
    ('age_18_24', None),
    (['gender'], FunctionTransformer(numpy.log))
])

classifier = VotingClassifier([
                                ('Forest', RandomForestClassifier()),
                                ('Regression', LogisticRegression()) 
                              ])

click_pipeline = PMMLPipeline([
    ('mapper', mapper),
 	('classifier', classifier)
 ])

click_pipeline = click_pipeline.fit(click_data.drop('click', 1), click_data.click)
sklearn2pmml(click_pipeline, "click2.pmml")

