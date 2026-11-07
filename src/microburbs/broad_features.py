"""Optional broad event attributes from the shared package, with causal joins."""
from pathlib import Path

import numpy as np
import pandas as pd

from .historical_attributes import AttributeEnhancer, build_snapshots

API_NUMERIC = ['bedrooms', 'bathrooms', 'car_spaces', 'land_size']
OFFICIAL_NUMERIC = ['bedrooms', 'bathrooms', 'car_spaces']
STRUCTURE_NUMERIC = ['land_size', 'floor', 'flats_in_building']
BROAD_NUMERIC = ([f'api_hist_{c}' for c in API_NUMERIC]
                 + [f'official_event_{c}' for c in OFFICIAL_NUMERIC]
                 + ['api_attribute_age_years', 'official_event_attribute_age_years']
                 + [f'ref_api_{c}' for c in STRUCTURE_NUMERIC]
                 + ['ref_official_flats_in_building'])
BROAD_CATEGORICAL = ['api_hist_property_type', 'ref_api_property_type',
                     'ref_api_dwelling_type', 'ref_api_level', 'ref_official_block_type']
RENT_NUMERIC = ['rent_weekly', 'rent_bedrooms', 'rent_bathrooms', 'rent_car_spaces',
                'rent_age_years', 'rent_listing_count', 'rent_previous_weekly', 'rent_change_ratio']
RENT_CATEGORICAL = ['rent_property_type']


def dated_states(frame, fields, *, official=False):
    """Preserve source separation; tied conflicting values remain unspecified."""
    s = frame[['gnaf_id', 'sold_date', *fields]].copy()
    s['available_date'] = pd.to_datetime(s.sold_date, errors='raise').astype('datetime64[ns]')
    if official:
        ad = pd.to_datetime(frame.advertised_date, errors='coerce').astype('datetime64[ns]')
        # A known later advertisement is usable only after it actually occurred.
        s['available_date'] = s.available_date.where(ad.isna() | ad.le(s.available_date), ad)
    for field in fields:
        if field != 'property_type':
            val = pd.to_numeric(s[field], errors='coerce')
            s[field] = val.where(np.isfinite(val) & val.ge(0))
        else:
            s[field] = s[field].astype('string').replace('', pd.NA)
    grouped = s.groupby(['gnaf_id', 'available_date'], sort=False)
    state = grouped[fields].first().where(grouped[fields].nunique().le(1)).reset_index()
    state = state.sort_values(['gnaf_id', 'available_date'], kind='stable')
    state[fields] = state.groupby('gnaf_id')[fields].ffill()
    return state


def match_states(states, rows):
    q = rows[['gnaf_id', 'as_of']].reset_index(drop=True).copy()
    q['as_of'] = pd.to_datetime(q.as_of).astype('datetime64[ns]')
    q['_order'] = np.arange(len(q))
    states = states.copy()
    q['gnaf_id'] = q.gnaf_id.astype(str)
    states['gnaf_id'] = states.gnaf_id.astype(str)
    return pd.merge_asof(q.sort_values('as_of', kind='stable'),
                         states.sort_values('available_date', kind='stable'),
                         left_on='as_of', right_on='available_date', by='gnaf_id',
                         direction='backward').sort_values('_order')


class BroadEnhancer:
    """Event-associated attributes and separately named stable-structure assumptions.

    API attributes use their sale row's event date. Official attributes without
    an ad date use the sale row date; a later explicit ad date delays availability.
    These are recorded assumptions in the v3 package, not publication-time claims.
    """
    def __init__(self, bundle, cutoff):
        bundle = Path(bundle)
        fold = bundle/'folds'/cutoff
        self.base = AttributeEnhancer(build_snapshots(pd.read_parquet(fold/'attribute_observations.parquet')))
        api = pd.read_parquet(fold/'api_sale_details.parquet')
        official = pd.read_parquet(fold/'official_sale_details.parquet')
        if api.sold_date.gt(cutoff).any() or official.sold_date.gt(cutoff).any():
            raise ValueError('Future sale observation in fold')
        self.api = dated_states(api, [*API_NUMERIC, 'property_type'])
        self.official = dated_states(official, OFFICIAL_NUMERIC, official=True)
        self.api_structure = pd.read_parquet(bundle/'references/api_structure.parquet').set_index('gnaf_id')
        self.official_structure = pd.read_parquet(bundle/'references/official_structure.parquet').set_index('gnaf_id')
        if self.api_structure.index.duplicated().any() or self.official_structure.index.duplicated().any():
            raise ValueError('Duplicate structural reference ID')

    def __call__(self, X, query_rows, history=None, cutoff=None, *, training=False):
        out = self.base(X, query_rows, history, cutoff, training=training)
        for states, fields, prefix, age_name in [
            (self.api, API_NUMERIC, 'api_hist_', 'api_attribute_age_years'),
            (self.official, OFFICIAL_NUMERIC, 'official_event_', 'official_event_attribute_age_years')]:
            m = match_states(states, query_rows)
            for field in fields:
                out[prefix+field] = m[field].to_numpy(float)
            out[age_name] = ((m.as_of-m.available_date).dt.days/365.25).to_numpy(float)
            if prefix == 'api_hist_':
                out['api_hist_property_type'] = m.property_type.fillna('').astype(str).to_numpy()
        ids = query_rows.gnaf_id.reset_index(drop=True)
        for c in STRUCTURE_NUMERIC:
            out['ref_api_'+c] = pd.to_numeric(ids.map(self.api_structure[c]), errors='coerce').to_numpy(float)
        for c in ['property_type', 'dwelling_type', 'level']:
            out['ref_api_'+c] = ids.map(self.api_structure[c]).fillna('').astype(str).to_numpy()
        out['ref_official_flats_in_building'] = pd.to_numeric(ids.map(self.official_structure.flats_in_building), errors='coerce').to_numpy(float)
        out['ref_official_block_type'] = ids.map(self.official_structure.block_type).fillna('').astype(str).to_numpy()
        return out


def load_bundle_enhancer(bundle, cutoff):
    fold = Path(bundle)/'folds'/cutoff
    if (fold/'rent_history.parquet').exists():
        return RentEnhancer(bundle, cutoff)
    if (fold/'api_sale_details.parquet').exists():
        return BroadEnhancer(bundle, cutoff)
    return AttributeEnhancer(build_snapshots(pd.read_parquet(fold/'attribute_observations.parquet')))


def rent_states(frame):
    """A model-specific daily view; the shared source keeps every advertisement."""
    fields = ['weekly_rent','bedrooms','bathrooms','car_spaces','property_type']
    s = frame.rename(columns={'gnaf_pid':'gnaf_id'})[['gnaf_id','listing_date',*fields]].copy()
    s['available_date'] = pd.to_datetime(s.listing_date,errors='raise').astype('datetime64[ns]')
    for c in fields[:-1]:
        v = pd.to_numeric(s[c],errors='coerce')
        s[c] = v.where(np.isfinite(v) & (v.gt(0) if c=='weekly_rent' else v.ge(0)))
    s['property_type'] = s.property_type.astype('string').replace('',pd.NA)
    groups = s.groupby(['gnaf_id','available_date'],sort=False)
    states = groups[fields].first().where(groups[fields].nunique().le(1))
    states['weekly_rent'] = groups.weekly_rent.median()
    states['listing_count'] = groups.size()
    states = states.reset_index().sort_values(['gnaf_id','available_date'],kind='stable')
    states[fields] = states.groupby('gnaf_id')[fields].ffill()
    states['listing_count'] = states.groupby('gnaf_id').listing_count.cumsum()
    states['previous_weekly'] = states.groupby('gnaf_id').weekly_rent.shift()
    states['change_ratio'] = states.weekly_rent/states.previous_weekly
    return states


class RentEnhancer(BroadEnhancer):
    def __init__(self,bundle,cutoff):
        super().__init__(bundle,cutoff)
        source = pd.read_parquet(Path(bundle)/'folds'/cutoff/'rent_history.parquet')
        if source.listing_date.gt(cutoff).any():raise ValueError('Future rental listing in fold')
        self.rents = rent_states(source)

    def __call__(self,X,query_rows,history=None,cutoff=None,*,training=False):
        out = super().__call__(X,query_rows,history,cutoff,training=training)
        m = match_states(self.rents,query_rows)
        for dest,source in [('weekly','weekly_rent'),('bedrooms','bedrooms'),('bathrooms','bathrooms'),
                            ('car_spaces','car_spaces'),('listing_count','listing_count'),
                            ('previous_weekly','previous_weekly'),('change_ratio','change_ratio')]:
            out['rent_'+dest] = m[source].to_numpy(float)
        out['rent_age_years'] = ((m.as_of-m.available_date).dt.days/365.25).to_numpy(float)
        out['rent_property_type'] = m.property_type.fillna('').astype(str).to_numpy()
        return out
