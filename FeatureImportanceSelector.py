import sys
import numpy as np
import pandas as pd

from pyspark import keyword_only
from pyspark.ml.base import Estimator
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.feature import VectorSlicer
from pyspark.ml.param.shared import HasOutputCol

def ExtractFeatureImp(featureImp, dataset, featuresCol):
    """
    Takes in a feature importance from a random forest / GBT model and map it to the column names
    Output as a pandas dataframe for easy reading
    
    rf = RandomForestClassifier(featuresCol="features")
    mod = rf.fit(train)
    ExtractFeatureImp(mod.featureImportances, train, "features")
    """
    
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))

class FeatureImpSelector(Estimator, HasOutputCol):
    """
    Uses feature importance score to select features for training
    Takes either the top n features or those above a certain threshold score
    estimator should either be a DecisionTreeClassifier, RandomForestClassifier or GBTClassifier
    featuresCol is inferred from the estimator
    """
    
    estimator = Param(Params._dummy(), "estimator", "estimator to be cross-validated")
    
    selectorType = Param(Params._dummy(), "selectorType",
                         "The selector type of the FeatureImpSelector. " +
                         "Supported options: numTopFeatures (default), threshold",
                         typeConverter=TypeConverters.toString)
    
    numTopFeatures = \
        Param(Params._dummy(), "numTopFeatures",
              "Number of features that selector will select, ordered by descending feature imp score. " +
              "If the number of features is < numTopFeatures, then this will select " +
              "all features.", typeConverter=TypeConverters.toInt)
    
    
    threshold = Param(Params._dummy(), "threshold", "The lowest feature imp score for features to be kept.",
                typeConverter=TypeConverters.toFloat)
    
    @keyword_only
    def __init__(self, estimator = None, selectorType = "numTopFeatures",
                 numTopFeatures = 20, threshold = 0.01, outputCol = "features"):
        
        super(FeatureImpSelector, self).__init__()
        self._setDefault(selectorType="numTopFeatures", numTopFeatures=20, threshold=0.01)
        kwargs = self._input_kwargs
        self._set(**kwargs)
        
    def setParams(self, estimator = None, selectorType = "numTopFeatures",
                 numTopFeatures = 20, threshold = 0.01, outputCol = "features"):
        """
        setParams(self, estimator = None, selectorType = "numTopFeatures",
                 numTopFeatures = 20, threshold = 0.01, outputCol = "features")
        Sets params for this ChiSqSelector.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setEstimator(self, value):
        """
        Sets the value of :py:attr:`estimator`.
        """
        return self._set(estimator=value)   
    
    def getEstimator(self):
        """
        Gets the value of estimator or its default value.
        """
        return self.getOrDefault(self.estimator)
    
    def setSelectorType(self, value):
        """
        Sets the value of :py:attr:`selectorType`.
        """
        return self._set(selectorType=value)

    def getSelectorType(self):
        """
        Gets the value of selectorType or its default value.
        """
        return self.getOrDefault(self.selectorType)
    
    def setNumTopFeatures(self, value):
        """
        Sets the value of :py:attr:`numTopFeatures`.
        Only applicable when selectorType = "numTopFeatures".
        """
        return self._set(numTopFeatures=value)

    def getNumTopFeatures(self):
        """
        Gets the value of numTopFeatures or its default value.
        """
        return self.getOrDefault(self.numTopFeatures)
    
    def setThreshold(self, value):
        """
        Sets the value of :py:attr:`Threshold`.
        Only applicable when selectorType = "threshold".
        """
        return self._set(threshold=value)

    def getThreshold(self):
        """
        Gets the value of threshold or its default value.
        """
        return self.getOrDefault(self.threshold)
    
    def _fit(self, dataset):

        est = self.getOrDefault(self.estimator)
        nfeatures = self.getOrDefault(self.numTopFeatures)
        threshold = self.getOrDefault(self.threshold)
        selectorType = self.getOrDefault(self.selectorType)
        outputCol = self.getOrDefault(self.outputCol)
        
        if ((est.__class__.__name__ != 'DecisionTreeClassifier') &
            (est.__class__.__name__ != 'DecisionTreeRegressor') &
            (est.__class__.__name__ != 'RandomForestClassifier') &
            (est.__class__.__name__ != 'RandomForestRegressor') &
            (est.__class__.__name__ != 'GBTClassifier') &
            (est.__class__.__name__ != 'GBTRegressor')):
            
            raise NameError("Estimator must be either DecisionTree, RandomForest or RandomForest Model")
            
        else:
        # Fit classifier & extract feature importance
        
            mod = est.fit(dataset)
            dataset2 = mod.transform(dataset)
            varlist = ExtractFeatureImp(mod.featureImportances, dataset2, est.getFeaturesCol())
        
            if (selectorType == "numTopFeatures"):
                varidx = [x for x in varlist['idx'][0:nfeatures]]
            elif (selectorType == "threshold"):
                varidx = [x for x in varlist[varlist['score'] > threshold]['idx']]
            else:
                raise NameError("Invalid selectorType")
            
        # Extract relevant columns
        return VectorSlicer(inputCol = est.getFeaturesCol(),
                           outputCol = outputCol,
                           indices = varidx)