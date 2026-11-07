"""Identical causal feature interface for all estimators. No target transaction attributes."""
from collections import defaultdict, deque
import numpy as np
import pandas as pd

CATEGORICAL = ['suburb','postcode','street','building_key','unit_suffix']
NUMERIC = ['unit_number','unit_hundreds_proxy','house_number_numeric','time_years',
           'prior_price','prior_age_years','prior_area_sqm','prior_city_index',
           'city_median','suburb_median','building_median','suburb_count','building_count',
           'building_observed_units','repeat_adjusted']
FEATURES = CATEGORICAL + NUMERIC

class HistoryFeatures:
    """fit(history); transform(queries with as_of) is shared across model routes.

    History includes only first-published, eligible records. At each query only
    available_date <= as_of enters any state. Parcel/area/purpose of the query's
    future transaction are NOT used. Unit-hundreds is a weak proxy, not actual floor.
    """
    def fit(self, history):
        self.history=history.sort_values(['available_date','contract_date','row_id']).copy()
        return self

    def transform(self, queries):
        q=queries.copy().reset_index(drop=True)
        q['_order']=np.arange(len(q)); q=q.sort_values('as_of')
        events=list(self.history.itertuples(index=False)); cursor=0
        histories=defaultdict(deque); observed=defaultdict(set); latest={}; cache={}
        # Index values themselves obey information availability. Historical index
        # is evaluated at the prior contract date, using only then-published data.
        ev=self.history[['available_date','price']].copy().sort_values('available_date')
        if len(ev):
            city=ev.set_index('available_date').price.rolling('365D').median()
            city=city[~city.index.duplicated(keep='last')]
        else: city=pd.Series(dtype=float,index=pd.DatetimeIndex([]))
        def index_at(date):
            i=city.index.searchsorted(date,side='right')-1
            return float(city.iloc[i]) if i>=0 else np.nan
        def stats(key,date,days):
            token=(key,date)
            if token in cache: return cache[token]
            h=histories[key]; oldest=date-pd.Timedelta(days=days)
            while h and h[0][0]<=oldest: h.popleft()
            vals=[v for _,v in h]
            out=(float(np.median(vals)) if vals else np.nan,len(vals))
            cache[token]=out
            return out
        out=[]
        for row in q.itertuples(index=False):
            date=pd.Timestamp(row.as_of)
            while cursor<len(events) and events[cursor].available_date<=date:
                e=events[cursor]; cursor+=1
                for key in [('city','all'),('suburb',e.suburb),('building',e.building_key)]:
                    histories[key].append((e.available_date,e.price))
                observed[e.building_key].add(e.property_key)
                old=latest.get(e.property_key)
                if old is None or e.contract_date>old.contract_date: latest[e.property_key]=e
            cm,_=stats(('city','all'),date,365)
            sm,sn=stats(('suburb',row.suburb),date,730)
            bm,bn=stats(('building',row.building_key),date,1095)
            old=latest.get(row.property_key)
            pp=old.price if old else np.nan
            age=(date-old.contract_date).days/365.25 if old else np.nan
            area=old.area_sqm if old else np.nan
            ix=index_at(old.contract_date) if old else np.nan
            adjusted=pp*np.clip(cm/ix,.5,2.) if old and np.isfinite(ix) and ix>0 else pp
            un=pd.to_numeric(pd.Series([row.unit]).str.extract(r'^(\d+)',expand=False),errors='coerce').iloc[0]
            hn=pd.to_numeric(pd.Series([row.house_number]).str.extract(r'^(\d+)',expand=False),errors='coerce').iloc[0]
            out.append(dict(suburb=row.suburb,postcode=row.postcode,street=row.street,
                building_key=row.building_key,unit_suffix=''.join(c for c in row.unit if c.isalpha()),
                unit_number=un,unit_hundreds_proxy=np.floor(un/100) if np.isfinite(un) else np.nan,
                house_number_numeric=hn,time_years=(date-pd.Timestamp('2010-01-01')).days/365.25,
                prior_price=pp,prior_age_years=age,prior_area_sqm=area,prior_city_index=ix,
                city_median=cm,suburb_median=sm,building_median=bm,suburb_count=sn,building_count=bn,
                building_observed_units=len(observed[row.building_key]),repeat_adjusted=adjusted))
        # q preserves original integer indices after sort; restore caller's order.
        result=pd.DataFrame(out,index=q.index).sort_index()[FEATURES]
        for c in CATEGORICAL: result[c]=result[c].fillna('').astype(str)
        return result
