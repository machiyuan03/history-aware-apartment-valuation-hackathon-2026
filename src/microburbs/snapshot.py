"""Cutoff-specific historical state: vectorized joins instead of per-target scans.

Snapshot contains no target data and can be built offline. Any history delta
invalidates aggregate state; refresh rebuilds compact state from legal history.
Existing per-training-row causal features remain reusable for unchanged sources.
"""
import numpy as np
import pandas as pd
from .features import FEATURES, CATEGORICAL

class FeatureSnapshot:
    def fit(self,history,cutoff):
        self.cutoff=pd.Timestamp(cutoff)
        h=history[(history.available_date<=self.cutoff)&(history.contract_date<=self.cutoff)].copy()
        if h.empty: raise ValueError('No legally available history')
        h=h.sort_values(['available_date','contract_date','row_id'])
        self.history_n=len(h)
        def aggregate(key,days):
            recent=h[h.available_date>self.cutoff-pd.Timedelta(days=days)]
            return recent.groupby(key).price.agg(['median','count'])
        self.suburb=aggregate('suburb',730);self.building=aggregate('building_key',1095)
        self.observed=h.groupby('building_key').property_key.nunique()
        recent=h[h.available_date>self.cutoff-pd.Timedelta(days=365)]
        self.city=float(recent.price.median())
        self.latest=h.sort_values(['contract_date','available_date','row_id'],ascending=[False,True,True],kind='stable').drop_duplicates('property_key',keep='first').set_index('property_key')
        ev=h.sort_values('available_date',kind='stable').set_index('available_date').price.rolling('365D').median()
        ev=ev[~ev.index.duplicated(keep='last')]
        ix=ev.index.searchsorted(self.latest.contract_date,side='right')-1
        self.latest['prior_city_index']=np.where(ix>=0,ev.to_numpy()[np.maximum(ix,0)],np.nan)
        return self

    def transform(self,queries):
        q=queries.reset_index(drop=True)
        if 'as_of' in q and not pd.to_datetime(q.as_of).eq(self.cutoff).all():
            raise ValueError('Snapshot cutoff differs from query as_of; rebuild required')
        z=pd.DataFrame(index=q.index)
        for c in ['suburb','postcode','street','building_key']: z[c]=q[c].fillna('').astype(str)
        z['unit_suffix']=q.unit.fillna('').str.replace(r'[^A-Za-z]','',regex=True)
        z['unit_number']=pd.to_numeric(q.unit.str.extract(r'^(\d+)',expand=False),errors='coerce')
        z['unit_hundreds_proxy']=np.floor(z.unit_number/100)
        z['house_number_numeric']=pd.to_numeric(q.house_number.str.extract(r'^(\d+)',expand=False),errors='coerce')
        z['time_years']=(self.cutoff-pd.Timestamp('2010-01-01')).days/365.25
        for dest,source in [('prior_price','price'),('prior_area_sqm','area_sqm'),('prior_city_index','prior_city_index')]:
            z[dest]=q.property_key.map(self.latest[source])
        dates=q.property_key.map(self.latest.contract_date)
        z['prior_age_years']=(self.cutoff-dates).dt.days/365.25
        z['city_median']=self.city
        for key,table in [('suburb',self.suburb),('building',self.building)]:
            col='suburb' if key=='suburb' else 'building_key'
            z[key+'_median']=q[col].map(table['median'])
            z[key+'_count']=q[col].map(table['count']).fillna(0).astype(int)
        z['building_observed_units']=q.building_key.map(self.observed).fillna(0).astype(int)
        ratio=(z.city_median/z.prior_city_index).clip(.5,2.)
        z['repeat_adjusted']=z.prior_price*ratio
        z['repeat_adjusted']=z.repeat_adjusted.fillna(z.prior_price)
        return z[FEATURES]
