import numpy as np
import pandas as pd

from microburbs.broad_features import dated_states, match_states, rent_states


def test_event_attributes_allow_missing_ad_but_delay_known_future_ad():
    source = pd.DataFrame({'gnaf_id':['A','A','B'], 'sold_date':['2020-01-01','2021-01-01','2020-01-01'],
                           'advertised_date':['','2021-03-01',''], 'bedrooms':['1','3','2']})
    states = dated_states(source, ['bedrooms'], official=True)
    rows = pd.DataFrame({'gnaf_id':['A','A','A','A'],
                         'as_of':['2019-12-31','2020-02-01','2021-02-01','2021-03-02']})
    actual = match_states(states, rows)
    assert np.isnan(actual.bedrooms.iloc[0])
    assert actual.bedrooms.iloc[1:].tolist() == [1,1,3]
    source.loc[1,'bedrooms']='999'
    changed = match_states(dated_states(source,['bedrooms'],official=True),rows)
    pd.testing.assert_series_equal(actual.bedrooms.iloc[:3],changed.bedrooms.iloc[:3])


def test_api_attributes_are_causal_and_keep_sources_out_of_model_targets():
    source = pd.DataFrame({'gnaf_id':['A','A','A'], 'sold_date':['2020-01-01','2021-01-01','2021-01-01'],
                           'bedrooms':[1,2,3], 'property_type':['Unit','Unit','Unit'], 'sold_price':[9,99,999]})
    state = dated_states(source,['bedrooms','property_type'])
    assert 'sold_price' not in state
    # Contradictory same-event values do not arbitrarily override an earlier observation.
    assert state.bedrooms.tolist() == [1,1]


def test_rental_features_exclude_later_ads_and_un_timestamped_lease_status():
    source=pd.DataFrame({'gnaf_pid':['A','A','A'],'listing_date':['2020-01-01','2021-01-01','2021-01-01'],
        'weekly_rent':[500,600,650],'bedrooms':[1,2,2],'bathrooms':[1,1,1],'car_spaces':[0,0,0],
        'property_type':['Unit']*3,'leased':[True,True,True]})
    q=pd.DataFrame({'gnaf_id':['A','A','A'],'as_of':['2019-12-31','2020-06-01','2021-02-01']})
    state=rent_states(source);out=match_states(state,q)
    assert 'leased' not in state
    assert np.isnan(out.weekly_rent.iloc[0])
    assert out.weekly_rent.iloc[1:].tolist()==[500,625]
    assert out.listing_count.iloc[2]==3 and out.previous_weekly.iloc[2]==500
    source.loc[1:,'weekly_rent']=9999
    changed=match_states(rent_states(source),q)
    pd.testing.assert_frame_equal(out.iloc[:2],changed.iloc[:2])
