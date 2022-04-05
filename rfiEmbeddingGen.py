# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:28:25 2022

@author: NCHAREST
"""
import json
import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.stats import spearmanr, pearsonr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

class RFICalculator:
    expAD = None
    isBinary = None

    ## Class containing functions to create an embedding for a data set using the importances of a random forest trained on the data
    def __init__(self):
        ## Basic Constructor
        self.df = None        
        pass
        
    def importSplit(self, tsv, yLabel, **kwargs):
        ## Import method designed to handle tsvs
        # self.dfFull = pd.read_csv(tsvPath, delimiter='\t', **kwargs)
        self.dfFull = tsv
        self.X_full = self.dfFull.drop(['ID', yLabel], axis=1)
        self.Y_full = self.dfFull[yLabel]

        if self.Y_full.isin([0, 1]).all():
            self.isBinary = True
        else:
            self.isBinary = False

        
        self.feature_names = list(self.X_full.columns)
        
    def basicClean(self):
        ## Rudimentary function to clean out null descriptor columns
        # self.zeroColumns = self.X_full.columns[(self.X_full==0).all()]
        # print(self.zeroColumns)
        self.X_full = self.X_full.loc[:, (self.X_full != 0).any(axis=0)]
        # self.X_full = self.X_full.drop(self.zeroColumns, axis=1)
        self.feature_names = list(self.X_full.columns)
        
    def wardsMethod(self, threshold):  
        ## This method implements Ward's hierarchical clustering on the distance matrix derived from Spearman's correlations between descriptors.
        ## Inputs: threshold (float) -- this is the cutoff t-value that determines the size and number of colinearity clusters.
        ########## test (Boolean) -- if True then trains a RF model with default hyperparameters using Ward embedding
        ## Output: sets self.wardsFeatures -- the list of features that have been identified as non-colinear.
        ## Source documentation: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
        
        ## We standardize data
        scaler = preprocessing.StandardScaler().fit(self.X_full)
        ## Compute spearman's r and ensure symmetry of correlation matrix
        corr = spearmanr(scaler.transform(self.X_full)).correlation
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1)
        ## Compute distance matrix and form hierarchical clusters
        distance_matrix = 1 - np.abs(corr)
        dist_linkage = hierarchy.ward(squareform(distance_matrix))        
        cluster_ids = hierarchy.fcluster(dist_linkage, threshold, criterion="distance")
        self.clusters = cluster_ids
        ## Pull out one representative descriptor from each cluster
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)            
        selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
        named_features = [self.feature_names[i] for i in selected_features]
        ## Set attribute with features that are not colinear
        self.wardsFeatures = [self.feature_names[i] for i in selected_features]
        
    def hyperparameterOptimizationRFRegressor(self, embedding, nEstimators, featuresPerTree, depthOfTrees):
        ## This is a method for optimizing the hyperparameters of a Random Forest model
        ## Inputs:
        ########## embedding (list of strings) -- the descriptor names for the embedding of the model
        ########## nEstimators (list of integers) -- range of number of trees per forest
        ########## featuresPerTree (list of integers, floats or supported strings) -- see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        ########## depthOfTrees (list of integers) -- maximum depths of trees to be considers
        ## Outputs:
        ########## sets optimizerResults attribute with best performing hyperparameters tested
            
        hyperParameterGrid = {'n_estimators' : nEstimators, 'max_features' : featuresPerTree, 'max_depth':depthOfTrees}
        scaler = preprocessing.StandardScaler().fit(self.X_full[embedding])
        X_train_scaled = scaler.transform(self.X_full[embedding])
        Y_train = self.Y_full
        
        rf = RandomForestRegressor()
        rfOptimizer = RandomizedSearchCV(estimator=rf, param_distributions=hyperParameterGrid, cv=5, random_state = 1117, n_iter = 25)
        rfOptimizer.fit(X_train_scaled, Y_train)
        
        self.optimizerResults = rfOptimizer.best_params_
         
     
    def randomForestImportanceRegressor(self, embedding, hyperparameters, numDesc):
        ### Trains a random forest regressor with the argued hyperparameters and returns the 12 most important descriptors based on mean decrease in impurity
        rfr = RandomForestRegressor(n_estimators = hyperparameters['n_estimators'], max_features = hyperparameters['max_features'], max_depth=hyperparameters['max_depth'], random_state = 924)
        scaler = preprocessing.StandardScaler().fit(self.X_full[embedding])
        
        X_train_scaled = scaler.transform(self.X_full[embedding])
        Y_train = self.Y_full
        
        rfr.fit(X_train_scaled, Y_train)
        
        importances = rfr.feature_importances_
        scoredFeatures = []
        
        for i in range(len(embedding)):
            scoredFeatures.append([embedding[i], importances[i]])
        
        scoredFeatures.sort(key=lambda x:x[1], reverse=True)
        
        mdiEmbedding = [i[0] for i in scoredFeatures[0:numDesc]]
        
        # return mdiEmbedding, scoredFeatures[0:numDesc]

        features = ''
        for i in range(numDesc):
            features = features + mdiEmbedding[i] + "\t"

        scoredfeatures = ''
        for i in range(numDesc):
            scoredfeatures = scoredfeatures + str(scoredFeatures[i][1]) + "\t"

        # return mdiEmbedding, scoredFeatures[0:numDesc]

        return features, scoredfeatures

    def randomForestImportanceClassifier(self, embedding, numDesc):
        ### Trains a random forest classifier and returns the 12 most important descriptors based on mean decrease in impurity

        rfc = RandomForestClassifier(random_state = 924)
        scaler = preprocessing.StandardScaler().fit(self.X_full[embedding])
        
        X_train_scaled = scaler.transform(self.X_full[embedding])
        Y_train = self.Y_full
        
        rfc.fit(X_train_scaled, Y_train)
        print(rfc.score(X_train_scaled, Y_train))
        
        importances = rfc.feature_importances_
        scoredFeatures = []
        
        for i in range(len(embedding)):
            scoredFeatures.append([embedding[i], importances[i]])
        
        scoredFeatures.sort(key=lambda x:x[1], reverse=True)
        
        mdiEmbedding = [i[0] for i in scoredFeatures[0:numDesc]]

        features = ''
        for i in range(numDesc):
            features = features + mdiEmbedding[i] + "\t"

        scoredfeatures = ''
        for i in range(numDesc):
            scoredfeatures = scoredfeatures + str(scoredFeatures[i][1]) + "\t"
        
        # return mdiEmbedding, scoredFeatures[0:numDesc]

        return features, scoredfeatures
      
    def findEmbedding(self, tsv, numDesc=12):
        ## Prime Method to execute RFI selection process for a regression problem
        self.importSplit(tsv, yLabel='Property')
        self.basicClean()
        if self.isBinary == False:
            self.wardsMethod(0.5)
            self.hyperparameterOptimizationRFRegressor(self.wardsFeatures, [120], [0.2, 0.4, 0.6, 0.8], [2,6,10])
            mdiEmbedding, mdiScored = self.randomForestImportanceRegressor(self.wardsFeatures, self.optimizerResults, numDesc)
            jsonObject = json.dumps({'embedding' : mdiEmbedding, 'importances' : mdiScored })
            print(jsonObject)
            return jsonObject
        if self.isBinary == True:
            self.wardsMethod(0.5)
            mdiEmbedding, mdiScored = self.randomForestImportanceClassifier(self.wardsFeatures, numDesc)
            jsonObject = json.dumps({'embedding' : mdiEmbedding, 'importances' : mdiScored })
            print(jsonObject)
            return jsonObject


if __name__ == '__main__':
    instance = RFICalculator()
    tsv = pd.read_csv(r"C:\Users\Weeb\Documents\QSARmod\data\DataSetsBenchmarkTEST_Toxicity\LLNA\LLNA_training_set-2d.csv", delimiter='\t')
    instance.findEmbedding(tsv)
