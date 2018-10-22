#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 21:59:35 2018

@author: edmund



Plan

LogD regression
Generally - logD = logP with sigmoidal correction for presence of charge (depending on microspecies proportion at a pH)

RDKit descriptors, normalised

pKA would be nice, but to my knowledge RDKit does not offer this.  I'm also only using open source software here
"""


##################
# imports
##################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
#from sklearn.feature_selection import chi2, SelectKBest
 

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2D



##################
# vars
##################

workdir = "/home/edmund/Documents/tests/az/"

desclist = [ d[0] for d in Descriptors._descList ]

mdc = MolecularDescriptorCalculator( desclist )


##################
# read
##################


dataset = pd.read_csv( workdir + "CHEMBL3301363_data.txt", sep="\t")
y = dataset.STANDARD_VALUE

# this dataset is really complex.  I only really care about the compound ID, experimental logD, and useful descriptors (like logP)

dataset = dataset.loc[ :, ["MOLREGNO", "STANDARD_VALUE", "ALOGP", "CANONICAL_SMILES" ] ]
#mols = []
descriptors = []

# this may take a few minutes...
for s in range( 0, len(dataset) ) :
    smiles = dataset.loc[ s, "CANONICAL_SMILES" ]
    mol = Chem.MolFromSmiles( smiles )
    mols.append( mol )
    
    descs = np.concatenate( (
            mdc.CalcDescriptors( mol ), 
            rdMolDescriptors.CalcAUTOCORR2D( mol ),
            [dataset.loc[ s, "ALOGP" ]]  # logP is a known important part of logD
    ) )
    
    descriptors.append( descs )
    
    #fp = np.zeros((1,))
    #DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048, useChirality=True), fp)
    
    
 
    
    
##################
# pre-processing
##################    
    

    
    
    
# replace missing values
imputer = Imputer(strategy="median")
X = imputer.fit_transform( descriptors, y=y)    


# feature scaling
sc = MinMaxScaler()
X = sc.fit_transform(X)


# training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)    


##################
# regression 
##################    

simple_regressor = LinearRegression()

simple_cv_scores = cross_val_score( simple_regressor, X_train, y_train, scoring="neg_mean_absolute_error", cv=5 )

simple_regressor.fit(X_train, y_train)
simple_regressor.score(X_test, y_test)  # r-squared

"""

Linear regression fails to capture anything useful.  However, likely because many descriptors are more important than others

"""

rf_regressor = RandomForestRegressor( n_estimators = 200, max_depth = 8, max_features = 0.5, random_state=0 )

rfr_cv_scores = cross_val_score( rf_regressor, X_train, y_train, scoring="neg_mean_absolute_error", cv=5, n_jobs=5 )

rf_regressor.fit(X_train, y_train)
rf_regressor.score(X_test, y_test)  # r-squared.  I got 0.55, a mediocre result (but better than linear)
y_pred = rf_regressor.predict(X_test)
mean_absolute_error( y_test, y_pred )

rf_regressor_fimps = rf_regressor.feature_importances_
top5 = np.flip( np.argsort( rf_regressor_fimps ) ) [0:5]


"""

Of note is that the first two indices correspond to both logPs.  The next two are numbers of carboxylic acids.
The fifth is a component of PEOE_VSA, a radial distribution function of VSA contribitions for a certain partial charge range

As I mentioned before - a pKa model would be ideal but the carboxylic acid count evidently somewhat compensates

"""



residuals_test =  np.abs( y_test - y_pred ) 
residuals_test_indices = residuals_test.index
residuals_test = np.array( residuals_test )
worst5 = np.flip( np.argsort( residuals_test ) )[ 0 : 5 ]
worst5 = residuals_test_indices[worst5]


worst5mols = []
for m in worst5:
    mol = mols[m]
    AllChem.Compute2DCoords(mol)
    worst5mols.append( mol )
    
Draw.MolsToImage(worst5mols)


"""

A look at these 5 worst outliers shows that these would normally be protonated in water under "standard" conditions.

Most have some form of strongly basic amine.  I don't think any of the descriptors account for these

In addition, the ChEMBL ALogP is usually about 3 units higher than the experimental, further supporting my charge argument

"""
