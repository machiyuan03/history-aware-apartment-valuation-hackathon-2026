import pandas as pd
import numpy as np
import pytest
from microburbs.historical_attributes import collect_observations, build_snapshots, AttributeEnhancer


def test_source_dates_gate_attributes_and_never_export_prices(tmp_path):
    p=tmp_path/'sales.csv'
    p.write_text('gnaf_id,sold_date,advertised_date,bedrooms,bathrooms,car_spaces,price,advertised_price\n'
                 'a,2020-01-02,2020-01-01,1,1,0,SECRET,SECRET\n'
                 'a,2020-02-02,2020-03-01,8,8,8,SECRET,SECRET\n'
                 'a,2026-06-28,2026-01-01,99,99,99,SEALED,SEALED\n')
    obs,audit=collect_observations(p)
    assert len(obs)==1 and audit['development_rows']==2
    assert 'price' not in ' '.join(obs.columns)
    assert not any('SECRET' in str(v) for v in obs.to_numpy().ravel())
    with pytest.raises(ValueError,match='sealed'):
        collect_observations(p,end='2026-08-31')


def test_features_are_asof_and_do_not_see_own_or_future_sale():
    obs=pd.DataFrame([
        dict(gnaf_id='a',available_date='2020-01-02',advertised_date='2020-01-01',bedrooms_raw='1',bathrooms_raw='1',car_spaces_raw='0'),
        dict(gnaf_id='a',available_date='2021-01-02',advertised_date='2021-01-01',bedrooms_raw='3',bathrooms_raw='',car_spaces_raw='1')])
    enhancer=AttributeEnhancer(build_snapshots(obs))
    q=pd.DataFrame({'gnaf_id':['a','a','a','b'],'as_of':['2020-01-01','2020-02-01','2021-01-03','2021-01-03'],
                    'address_level':['LEVEL 2','LEVEL 2','LEVEL 2','BASEMENT 1'], 'address_unit_type':['UNIT']*4})
    X=pd.DataFrame({'city_median':[1.]*4}, index=[5,4,3,2])
    out=enhancer(X,q)
    assert out.index.equals(X.index)
    assert np.isnan(out.hist_bedrooms.iloc[0])
    assert out.hist_bedrooms.iloc[1:3].tolist()==[1,3]
    assert out.hist_bathrooms.iloc[2]==1
    assert out.hist_car_spaces.iloc[1]==0
    assert np.isnan(out.hist_bedrooms.iloc[3])
    assert out.parsed_level.tolist()==[2,2,2,-1]
    obs.loc[1,'bedrooms_raw']='999'
    changed=AttributeEnhancer(build_snapshots(obs))(X,q)
    assert changed.iloc[:2].equals(out.iloc[:2])


def test_disagreement_is_not_arbitrarily_averaged():
    obs=pd.DataFrame([dict(gnaf_id='a',available_date='2020-01-02',advertised_date='2020-01-01',bedrooms_raw=str(b),bathrooms_raw='1',car_spaces_raw='0') for b in [1,3]])
    s=build_snapshots(obs)
    assert len(s)==1 and np.isnan(s.bedrooms.iloc[0]) and s.bathrooms.iloc[0]==1


def test_query_seconds_and_source_nanoseconds_have_same_asof_result():
    obs=pd.DataFrame([dict(gnaf_id='a',available_date='2020-01-02',advertised_date='2020-01-01',bedrooms_raw='1',bathrooms_raw='1',car_spaces_raw='0')])
    e=AttributeEnhancer(build_snapshots(obs))
    q=pd.DataFrame({'gnaf_id':['a'],'as_of':pd.Series(['2020-02-01'],dtype='datetime64[s]')})
    assert e(pd.DataFrame({'placeholder':[1]}),q).hist_bedrooms.iloc[0]==1
