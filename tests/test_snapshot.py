import numpy as np
import pandas as pd
import pytest
from microburbs.snapshot import FeatureSnapshot
from microburbs.features import HistoryFeatures
from test_protocol import history

def test_snapshot_equal_to_causal_reference_and_handles_unknowns():
    h=history();q=h.copy();q['as_of']=pd.Timestamp('2020-04-15')
    extra=q.iloc[[0]].copy();extra['property_key']='UNKNOWN';extra['building_key']='NEW';extra['suburb']='NEW'
    q=pd.concat([q,extra],ignore_index=True)
    a=HistoryFeatures().fit(h).transform(q);b=FeatureSnapshot().fit(h,'2020-04-15').transform(q)
    pd.testing.assert_frame_equal(a,b,check_dtype=False)

def test_snapshot_rejects_other_cutoff_and_future_history():
    h=history();q=h.iloc[[0]].copy();q['as_of']=pd.Timestamp('2020-04-15')
    a=FeatureSnapshot().fit(h,'2020-04-15').transform(q)
    h.loc[h.available_date>'2020-04-15','price']=99999999
    pd.testing.assert_frame_equal(a,FeatureSnapshot().fit(h,'2020-04-15').transform(q))
    with pytest.raises(ValueError): FeatureSnapshot().fit(h,'2020-04-14').transform(q)

def test_same_contract_date_keeps_first_publicly_seen_price():
    h=history();repeat=h.iloc[[0]].copy();repeat['available_date']=pd.Timestamp('2020-03-10');repeat['price']=990000;repeat['row_id']='restated'
    h=pd.concat([h,repeat],ignore_index=True);q=h.iloc[[0]].copy();q['as_of']=pd.Timestamp('2020-04-15')
    a=HistoryFeatures().fit(h).transform(q);b=FeatureSnapshot().fit(h,'2020-04-15').transform(q)
    pd.testing.assert_frame_equal(a,b,check_dtype=False)
