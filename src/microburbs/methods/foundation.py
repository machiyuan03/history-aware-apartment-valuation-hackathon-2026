"""Explicit-feature foundation regressors with cutoff-selected frozen contexts."""
import os
os.environ.setdefault('DO_NOT_TRACK','1')
import numpy as np
import pandas as pd
from .common import MethodBase
from ..feature_schema import FeatureSchema, positive_targets
from ..paths import data_root


class StableEncoder:
    def fit(self,X):
        self.schema=FeatureSchema();z=self.schema.fit_transform(X)
        self.center=z[self.schema.numeric].median().fillna(0)
        self.scale=(z[self.schema.numeric].quantile(.75)-z[self.schema.numeric].quantile(.25)).replace(0,1).fillna(1)
        self.cats={c:{v:i for i,v in enumerate(sorted(z[c].unique()))} for c in self.schema.categorical}
        self.categorical_indices=list(range(2*len(self.schema.numeric),2*len(self.schema.numeric)+len(self.cats)))
        return self
    def transform(self,X):
        z=self.schema.transform(X);raw=z[self.schema.numeric]
        numeric=((raw-self.center)/self.scale).clip(-30,30).fillna(0).to_numpy(float)
        categorical=np.column_stack([z[c].map(v).fillna(-1).to_numpy(float) for c,v in self.cats.items()])
        return np.c_[numeric,raw.isna().to_numpy(float),categorical].astype(np.float32)


class FoundationMethod(MethodBase):
    kind=None
    def __init__(self,context_rows=2048,batch_rows=128,n_estimators=2,
                 target_mode='city_normalized',context_strategy='recent_global',
                 recent_fraction=.5,seed=20261107,fit_mode='fit_with_cache'):
        if target_mode not in {'city_normalized','log_absolute'}:raise ValueError('Unknown target_mode')
        if context_strategy not in {'uniform','recent_global'}:raise ValueError('Unknown context_strategy')
        if min(context_rows,batch_rows,n_estimators)<1 or not 0<=recent_fraction<=1:raise ValueError('Invalid context options')
        self.context_rows=context_rows;self.batch_rows=batch_rows;self.n_estimators=n_estimators
        self.target_mode=target_mode;self.context_strategy=context_strategy;self.recent_fraction=recent_fraction
        self.seed=seed;self.fit_mode=fit_mode;self.calibration=0.;self.cache={}
        self.cache_stats={'queries':0,'hits':0,'miss_vectors':0}
    def select_context(self,X,context):
        allowed=np.asarray(context.get('context_eligible',np.ones(len(X),dtype=bool)))
        if allowed.shape!=(len(X),) or allowed.dtype!=bool:raise ValueError('Invalid context eligibility')
        eligible=np.flatnonzero(allowed)
        if not len(eligible):raise ValueError('No eligible context')
        n=min(self.context_rows,len(eligible));rng=np.random.default_rng(self.seed)
        if self.context_strategy=='uniform':return rng.choice(eligible,n,replace=False)
        recent_n=int(n*self.recent_fraction)
        recent=eligible[np.argsort(X.iloc[eligible].time_years.to_numpy(float),kind='stable')[-recent_n:]] if recent_n else np.array([],dtype=int)
        remainder=np.setdiff1d(eligible,recent)
        return np.r_[recent,rng.choice(remainder,n-len(recent),replace=False)]
    def base(self,X):
        if self.target_mode=='log_absolute':return np.ones(len(X))
        return X.city_median.where(X.city_median>0).fillna(self.city_fallback).to_numpy(float)
    def prepare(self,X,y,context):
        import torch
        torch.set_num_threads(4)
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        y=positive_targets(y,len(X));ix=self.select_context(X,context)
        self.encoder=StableEncoder().fit(X.iloc[ix]);z=self.encoder.transform(X.iloc[ix])
        city=X.city_median.where(X.city_median>0).median()
        self.city_fallback=float(city) if np.isfinite(city) else float(np.median(y))
        target=np.log(y[ix]/self.base(X.iloc[ix]))
        folder=data_root()/'models/foundation'/self.kind
        if self.kind != 'tabicl': raise ValueError('Only the final TabICL member is supported')
        from tabicl import TabICLRegressor
        self.model=TabICLRegressor(n_estimators=self.n_estimators,device='cuda',kv_cache=True,n_jobs=4,
            model_path=folder/'tabicl-regressor-v2-20260212.ckpt',allow_auto_download=False,
            random_state=self.seed,use_amp=True,verbose=False)
        self.model.fit(z,target);self.cache={};self.context_size=len(ix)
        self.context_positions=ix;self.generation=getattr(self,'generation',0)+1
        self.cache_stats={'queries':0,'hits':0,'miss_vectors':0}
        self.train_X=X.reset_index(drop=True).copy();self.train_y=y.copy()
        return self
    def adapt(self,X,y,context):
        # The existing online interface supplies an increment. Rebuild context and
        # encoder from prior legal training plus that increment, never intercept-only.
        xx=pd.concat([self.train_X,X],ignore_index=True)
        yy=np.r_[self.train_y,positive_targets(y,len(X))]
        ctx=dict(context);ctx.pop('context_eligible',None)
        return self.prepare(xx,yy,ctx)
    def predict(self,X):
        z=self.encoder.transform(X);keys=[r.tobytes() for r in z]
        missing={k:i for i,k in enumerate(keys) if k not in self.cache}
        hits=sum(k in self.cache for k in keys);todo=list(missing.items())
        for start in range(0,len(todo),self.batch_rows):
            batch=todo[start:start+self.batch_rows];ix=[i for _,i in batch]
            pred=np.asarray(self.model.predict(z[ix]),dtype=float)
            self.cache.update({k:float(p) for (k,_),p in zip(batch,pred)})
        self.cache_stats['queries']+=len(X);self.cache_stats['hits']+=hits;self.cache_stats['miss_vectors']+=len(missing)
        return self.finish(np.exp(np.array([self.cache[k] for k in keys]))*self.base(X))
    def precompute(self,X):
        self.predict(X);return {'cached_vectors':len(self.cache),'input_rows':len(X)}
