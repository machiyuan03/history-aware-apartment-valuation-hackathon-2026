"""LightGBM log-price member of the final ensemble."""
import numpy as np
import pandas as pd
from .feature_schema import FeatureSchema, positive_targets, training_weights

class BoostModel:
    def __init__(self,kind,threads=4,seed=20261107,params=None):
        if kind != 'lightgbm': raise ValueError(f'Unknown boost kind: {kind}')
        self.kind=kind; self.threads=threads; self.seed=seed; self.params=dict(params or {})
        # Recency weighting is applied before fitting, outside the estimator.
        self.half_life_years=self.params.pop('half_life_years',None)
    def encode(self,X,fit=False):
        if fit: self.schema=FeatureSchema().fit(X)
        z=self.schema.transform(X)
        if fit: self.categories={c:list(z[c].unique()) for c in self.schema.categorical}
        for c in self.schema.categorical: z[c]=pd.Categorical(z[c],categories=self.categories[c])
        return z
    def fit(self,X,y,sample_weight=None):
        z=self.encode(X,fit=True)
        y=positive_targets(y,len(X))
        sample_weight=training_weights(z,self.half_life_years,sample_weight)
        from lightgbm import LGBMRegressor
        self.model=LGBMRegressor(n_estimators=1200,num_leaves=47,learning_rate=.03,
            min_child_samples=35,reg_lambda=7,n_jobs=self.threads,random_state=self.seed,
            objective='regression_l1',verbosity=-1,deterministic=True,force_col_wise=True)
        if self.params: self.model.set_params(**self.params)
        self.model.fit(z,np.log(y),sample_weight=sample_weight)
        return self
    def predict(self,X):
        return np.exp(self.model.predict(self.encode(X)))
