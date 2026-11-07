"""Dated, prior-sale attribute observations; never reads price fields."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd


FIELDS = ('bedrooms', 'bathrooms', 'car_spaces')
DEVELOPMENT_END = '2026-06-27'


def collect_observations(path, end=DEVELOPMENT_END):
    if end > DEVELOPMENT_END:
        raise ValueError('Development attribute preparation cannot cross sealed boundary')
    records = []
    audit = dict(development_rows=0, rows_with_attributes=0, eligible_observations=0,
                 missing_or_invalid_ad_date=0, ad_after_sale=0)
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'gnaf_id', 'sold_date', 'advertised_date', *FIELDS}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Required attribute/date columns missing')
        for index, row in enumerate(reader):
            sale = row['sold_date']
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', sale):
                raise ValueError('Invalid source sale date')
            if sale > end:
                continue  # Do not access this row's attributes, ads or price.
            audit['development_rows'] += 1
            raw = {key: row[key].strip() for key in FIELDS}
            if not any(raw.values()):
                continue
            audit['rows_with_attributes'] += 1
            ad = row['advertised_date'].strip()
            parsed = pd.to_datetime(ad, errors='coerce') if ad else pd.NaT
            if pd.isna(parsed) or parsed < pd.Timestamp('1990-01-01'):
                audit['missing_or_invalid_ad_date'] += 1
                continue
            ad = str(parsed.date())
            if ad > sale:
                audit['ad_after_sale'] += 1
                continue
            records.append({'gnaf_id': row['gnaf_id'], 'available_date': sale,
                            'advertised_date': ad, 'source_record_index': index,
                            **{key+'_raw': value for key, value in raw.items()}})
    columns = ['gnaf_id', 'available_date', 'advertised_date', 'source_record_index'] + [k+'_raw' for k in FIELDS]
    observations = pd.DataFrame(records, columns=columns)
    audit['eligible_observations'] = len(observations)
    return observations, audit


def build_snapshots(observations):
    """Optional last-known view; each source row stays available separately."""
    obs = observations.copy()
    for key in FIELDS:
        values = pd.to_numeric(obs[key+'_raw'], errors='coerce')
        obs[key] = values.where(np.isfinite(values) & values.ge(0))
    # Prefer the latest explicitly dated advertisement within the same sale event.
    latest_ad = obs.groupby(['gnaf_id', 'available_date']).advertised_date.transform('max')
    chosen = obs.loc[obs.advertised_date.eq(latest_ad)]
    group = chosen.groupby(['gnaf_id', 'available_date'], sort=False)
    values = group[list(FIELDS)].first()
    distinct = group[list(FIELDS)].nunique()
    values = values.where(distinct.le(1))  # Conflicting field remains unknown.
    values = values.reset_index().sort_values(['gnaf_id', 'available_date'], kind='stable')
    for key in FIELDS:
        values[key+'_observed_date'] = values.available_date.where(values[key].notna())
        values[[key, key+'_observed_date']] = values.groupby('gnaf_id')[[key, key+'_observed_date']].ffill()
    values['available_date'] = pd.to_datetime(values.available_date)
    return values.reset_index(drop=True)


class AttributeEnhancer:
    def __init__(self, snapshots):
        self.snapshots = snapshots.copy()
        self.snapshots['available_date'] = pd.to_datetime(self.snapshots.available_date).astype('datetime64[ns]')

    def __call__(self, X, query_rows, history=None, cutoff=None, *, training=False):
        if len(X) != len(query_rows):
            raise ValueError('Feature/query length mismatch')
        if not {'gnaf_id', 'as_of'} <= set(query_rows):
            raise ValueError('Dated GNAF queries required')
        out = X.copy()
        q = query_rows[['gnaf_id', 'as_of']].reset_index(drop=True).copy()
        q['as_of'] = pd.to_datetime(q.as_of).astype('datetime64[ns]')
        q['_order'] = np.arange(len(q))
        if q.as_of.isna().any():
            raise ValueError('Missing feature as_of')
        if len(self.snapshots):
            match = pd.merge_asof(q.sort_values('as_of', kind='stable'),
                self.snapshots.sort_values('available_date', kind='stable'),
                left_on='as_of', right_on='available_date', by='gnaf_id', direction='backward')
            match = match.sort_values('_order', kind='stable')
        else:
            match = q.copy()
            for key in FIELDS:
                match[key] = np.nan
                match[key+'_observed_date'] = pd.NaT
        observation_dates = []
        for key in FIELDS:
            out['hist_'+key] = match[key].to_numpy(float)
            observation_dates.append(pd.to_datetime(match[key+'_observed_date']))
        most_recent = pd.concat(observation_dates, axis=1).max(axis=1)
        out['attribute_age_years'] = ((match.as_of-most_recent).dt.days / 365.25).to_numpy(float)
        out['attribute_known'] = match[list(FIELDS)].notna().sum(axis=1).to_numpy(float)
        levels = query_rows.get('address_level', pd.Series('', index=query_rows.index)).fillna('').astype(str)
        parsed = pd.to_numeric(levels.str.extract(r'(-?\d+)\s*$', expand=False), errors='coerce')
        parsed = parsed.mask(levels.str.contains('LOWER GROUND|BASEMENT', case=False, regex=True), -1.)
        out['parsed_level'] = parsed.to_numpy(float)
        out['address_unit_type'] = query_rows.get('address_unit_type', pd.Series('', index=query_rows.index)).fillna('').astype(str).to_numpy()
        if {'city_median','prior_city_index','prior_price'} <= set(out):
            ratio = out.city_median / out.prior_city_index.replace(0,np.nan)
            out['repeat_adjusted_unclipped'] = (out.prior_price * ratio).replace([np.inf,-np.inf],np.nan).fillna(out.prior_price)
        return out

