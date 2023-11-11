import pandas as pd
import statsmodels.api as sm
from typing import List


class FeatureReduction(object):
    def __init__(self):
        pass

    @staticmethod
    def forward_selection(
        data: pd.DataFrame, target: pd.Series, significance_level: float = 0.05
    ) -> List[str]:  # 9 pts
        """
        Args:
            data: (pandas data frame) contains the feature matrix
            target: (pandas series) represents target feature to search to generate significant features
            significance_level: (float) threshold to reject the null hypothesis
        Return:
            forward_list: (python list) contains significant features. Each feature
            name is a string
        """
        sigfig = []
        features = data.columns
        
        while len(features) > 0:

            pVal = []
            for ft in features:
                X = data[[ft] + sigfig]
                X = sm.add_constant(X) 

                model = sm.OLS(target, X).fit()
                pVal.append((ft, model.pvalues[ft]))
            bestFeat, bestPVal = sorted(pVal, key=lambda x: x[1])[0]
            #print(pVal)
            if significance_level> bestPVal :
                features = [feat for feat in features if feat != bestFeat]
                sigfig.append(bestFeat)
            else: break
        
        return sigfig

    @staticmethod
    def backward_elimination(
        data: pd.DataFrame, target: pd.Series, significance_level: float = 0.05
    ) -> List[str]:  # 9 pts
        """
        Args:
            data: (pandas data frame) contains the feature matrix
            target: (pandas series) represents target feature to search to generate significant features
            significance_level: (float) threshold to reject the null hypothesis
        Return:
            backward_list: (python list) contains significant features. Each feature
            name is a string
        """

        ft = data.columns.tolist()
        while len(ft) > 0:
            pval = sm.OLS(target, sm.add_constant(data[ft])).fit().pvalues[1:]
            maxP = pval.max()
            if maxP > significance_level:
                rmFt = ft[pval.argmax()]
                ft.remove(rmFt)
            else: break
        return ft

        raise NotImplementedError
