import numpy as np
import pandas as pd
import pytest
from microburbs.features import HistoryFeatures
from microburbs.prepare import address_keys
from microburbs.submission import adapt_input,adapt_training,validate_submission

# Tiny constructed records test invariants only, never used for benchmark scores.
def history():
    d=pd.DataFrame([
      dict(row_id='a',unit='1',house_number='10',street='KING ST',suburb='SYDNEY',postcode='2000',contract_date='2020-01-01',available_date='2020-03-01',price=500000,area_sqm=50),
      dict(row_id='b',unit='2',house_number='10',street='KING ST',suburb='SYDNEY',postcode='2000',contract_date='2020-02-01',available_date='2020-04-01',price=800000,area_sqm=80),
      dict(row_id='c',unit='1',house_number='10',street='KING ST',suburb='SYDNEY',postcode='2000',contract_date='2021-01-01',available_date='2021-03-01',price=900000,area_sqm=55)])
    for c in ['contract_date','available_date']: d[c]=pd.to_datetime(d[c])
    return address_keys(d)

def test_only_public_information_and_no_target_area():
    h=history(); q=h.iloc[[0]].copy(); q['as_of']=pd.Timestamp('2020-03-15')
    x=HistoryFeatures().fit(h).transform(q)
    assert x.prior_price.iloc[0]==500000
    assert x.building_count.iloc[0]==1
    altered=h.copy(); altered.loc[1:,'price']=99999999; altered.loc[1:,'area_sqm']=99999
    pd.testing.assert_frame_equal(x,HistoryFeatures().fit(altered).transform(q))
    q['price']=1; q['area_sqm']=9999
    pd.testing.assert_frame_equal(x,HistoryFeatures().fit(h).transform(q))

def test_own_sale_not_known_before_contract():
    h=history(); q=h.iloc[[0]].copy(); q['as_of']=pd.Timestamp('2019-12-31')
    x=HistoryFeatures().fit(h).transform(q)
    assert np.isnan(x.prior_price.iloc[0]) and x.building_count.iloc[0]==0

def test_future_record_cannot_change_prior_city_adjustment():
    h=history(); q=h.iloc[[2]].copy(); q['as_of']=pd.Timestamp('2021-04-01')
    future=h.iloc[[1]].copy(); future['available_date']=pd.Timestamp('2022-01-01'); future['price']=999999999
    pd.testing.assert_frame_equal(HistoryFeatures().fit(h).transform(q),HistoryFeatures().fit(pd.concat([h,future])).transform(q))




def test_input_and_submission():
    d=pd.DataFrame([dict(gnaf_id='a',address='Unit 315, 2-4 Powell Street',suburb='Waterloo (NSW)',postcode='2017',unit_number='315')])
    a,q=adapt_input(d,expected_rows=1)
    assert a.property_key.iloc[0]=='315|2-4|POWELL ST|WATERLOO|2017'
    assert validate_submission(d.gnaf_id,pd.DataFrame({'gnaf_id':['a'],'value':[688000]}),1)['coverage']==1
    for ids,values in [(['b'],[500000]),(['a'],[np.nan]),(['a'],[0]),(['a'],[123.5]),(['a','a'],[1,2])]:
        with pytest.raises(ValueError): validate_submission(d.gnaf_id,pd.DataFrame({'gnaf_id':ids,'value':values}),1)

def test_official_training_refuses_post_cutoff():
    inp=pd.DataFrame([dict(gnaf_id='a',address='1/10 King Street',suburb='Sydney',postcode='2000',unit_number='1')])
    targets,_=adapt_input(inp,1)
    raw=pd.DataFrame({'id':['a'],'price':[500000],'date':['2026-09-01'],'published':['2026-10-01']})
    mapping={'gnaf_id':'id','price':'price','contract_date':'date','available_date':'published'}
    with pytest.raises(ValueError,match='after valuation cutoff'): adapt_training(raw,targets,mapping)

def test_official_adapter_converts_csv_area_strings():
    targets,_=adapt_input(pd.DataFrame([dict(gnaf_id='a',address='Unit 1, 10 King Street',suburb='Sydney',postcode='2000',unit_number='1')]),1)
    raw=pd.DataFrame({'gnaf_id':['a'],'price':['500000'],'contract_date':['2020-01-01'],'available_date':['2020-03-01'],'area_sqm':['50']})
    adapted,_=adapt_training(raw,targets,{c:c for c in raw.columns})
    assert adapted.area_sqm.dtype.kind in 'fi' and adapted.area_sqm.iloc[0]==50



def test_shared_candidate_prediction_contract():
    from microburbs.methods.common import prediction_vector
    np.testing.assert_array_equal(prediction_vector([100.,200.],2),[100.,200.])
    for malformed in [100., [[100.],[200.]], [100.], [100.,np.nan], [100.,0.]]:
        with pytest.raises(ValueError):prediction_vector(malformed,2)
