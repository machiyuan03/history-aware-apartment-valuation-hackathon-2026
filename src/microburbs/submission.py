"""Official schema boundary. No assumptions about unpublished training column names."""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .paths import dump_json
from .prepare import address_keys

OFFICIAL_COLUMNS=['gnaf_id','address','suburb','postcode','mb','block_type','latitude','longitude',
                  'unit_number','bedrooms','bathrooms','car_spaces','flats_in_building']

_UNIT_TYPES = ('UNIT', 'FLAT', 'APARTMENT', 'SHOP', 'OFFICE', 'SUITE', 'CARSPACE',
               'STORE', 'STUDIO', 'KIOSK', 'TOWNHOUSE', 'ROOM', 'PENTHOUSE',
               'GARAGE', 'REAR', 'HOUSE', 'SITE', 'SHED', 'DUPLEX', 'WAREHOUSE', 'STALL')
_LEVEL_TYPES = ('LOWER GROUND FLOOR', 'LEVEL', 'FLOOR', 'BASEMENT', 'MEZZANINE')
_COMPONENT = re.compile(r'^(' + '|'.join(_LEVEL_TYPES + _UNIT_TYPES)
                        + r')\s+([0-9A-Z]+)(?:\s*,\s*|\s+)(.+)$')
_STREET_BODY = re.compile(r'^([0-9]+[A-Z]?(?:\s*-\s*[0-9]+[A-Z]?)?)\s+([^,/]+)$')


def _address_components(address):
    """Remove only recognized subpremise prefixes, retaining their identity."""
    body = re.sub(r'\s+', ' ', address.upper().strip())
    unit, unit_type, level = '', '', ''
    while match := _COMPONENT.fullmatch(body):
        kind, number, body = match.groups()
        if kind in _LEVEL_TYPES:
            if level:
                return None
            level = kind + ' ' + number
        else:
            if unit:
                return None
            unit, unit_type = number, kind
    if slash := re.fullmatch(r'([0-9A-Z]+)\s*/\s*(.+)', body):
        if unit:
            return None
        unit, body = slash.groups()
        unit_type = 'UNIT'
    street = _STREET_BODY.fullmatch(body)
    if street is None or not re.search('[A-Z]', street[2]):
        return None
    return unit, unit_type, level, re.sub(r'\s+', '', street[1]), street[2].strip()


def _unit_identifier(value):
    value = re.sub(r'\s+', ' ', value.upper().strip())
    for kind in _UNIT_TYPES:
        if value.startswith(kind + ' '):
            return value[len(kind) + 1:].replace(' ', '')
    return value.replace(' ', '')

def validate_input(d,expected_rows=181711):
    required={'gnaf_id','address','suburb','postcode','unit_number'}
    if missing:=required-set(d.columns): raise ValueError(f'Missing input columns: {sorted(missing)}')
    ids=d.gnaf_id.fillna('').astype(str)
    if expected_rows is not None and len(d)!=expected_rows: raise ValueError(f'Expected {expected_rows} input rows, got {len(d)}')
    if ids.str.strip().eq('').any() or ids.duplicated().any(): raise ValueError('Input IDs must be nonempty and unique')
    return {'n':len(d),'missing_optional_columns':sorted(set(OFFICIAL_COLUMNS)-set(d.columns)),
            'missing_fraction':d.isna().mean().to_dict()}

def adapt_input(d,expected_rows=181711):
    quality=validate_input(d,expected_rows)
    d=d.fillna('').astype(str).copy(); d['unit']=d.unit_number.str.strip().str.replace(r'\.0$','',regex=True)
    house=[]; street=[]; units=[]; kinds=[]; levels=[]; conflicts=[]; failed=0; inferred=0
    for row in d.itertuples(index=False):
        parsed = _address_components(row.address)
        unit, kind, level, conflict = row.unit, '', '', False
        if parsed is None:
            house.append('UNPARSED_'+row.gnaf_id); street.append(''); failed+=1
        else:
            explicit, kind, level, number, road = parsed
            house.append(number); street.append(road)
            if not unit and explicit:
                unit = explicit if kind in {'UNIT', 'FLAT', 'APARTMENT'} else kind + ' ' + explicit
                inferred += 1
            elif unit and explicit:
                conflict = _unit_identifier(unit) != explicit
        units.append(unit); kinds.append(kind); levels.append(level); conflicts.append(conflict)
    d['unit']=units; d['house_number']=house; d['street']=street
    d['address_unit_type']=kinds; d['address_level']=levels; d['unit_number_conflict']=conflicts
    d=address_keys(d); d['row_id']=d.gnaf_id
    # Building membership may be shared. Distinct G-NAF IDs never acquire a
    # shared property identity merely because their visible address is equal.
    d['address_property_key']=d.property_key
    with_level=d.address_level.ne('')
    d.loc[with_level,'address_property_key'] += '|'+d.loc[with_level,'address_level']
    d['property_key']=d.address_property_key
    collision=d.property_key.duplicated(keep=False)
    d['address_key_collision']=collision
    unparsed=d.house_number.str.startswith('UNPARSED_')
    independent=collision | d.unit_number_conflict | unparsed
    d['gnaf_property_key']='GNAF|'+d.gnaf_id
    d.loc[independent,'property_key']=d.loc[independent,'gnaf_property_key']
    d.loc[unparsed,'building_key']='UNPARSED_'+d.loc[unparsed,'gnaf_id']
    if d.property_key.duplicated().any():
        raise ValueError('Distinct input IDs must retain distinct property keys')
    quality['unit_inferred_from_address']=inferred
    quality['address_parse_failures']=failed
    quality['parsed_level_rows']=int(with_level.sum())
    quality['unit_number_conflicts']=int(d.unit_number_conflict.sum())
    quality['address_key_collision_rows']=int(collision.sum())
    quality['address_key_collision_groups']=int(d.loc[collision,'address_property_key'].nunique())
    quality['independent_gnaf_property_keys']=int(independent.sum())
    quality['attributes_used']='Address fields only; current bedroom/bathroom/parking fields excluded from v1.'
    return d,quality

def adapt_training(raw,targets,mapping,cutoff='2026-08-31',*,deduplicate=True):
    """mapping maps canonical names to actual supplied column names.

    Time columns follow the caller's confirmed source-time semantics, including
    an explicitly authorized event-time protocol. A contract date does not by
    itself establish publication time; do not invent that interpretation.
    deduplicate selects an earliest-version model view; False retains linked rows.
    """
    required={'gnaf_id','price','contract_date','available_date'}
    if not required<=set(mapping): raise ValueError(f'Mapping must include {required}')
    if missing:=set(mapping.values())-set(raw.columns): raise ValueError(f'Mapped columns missing: {missing}')
    d=raw.rename(columns={v:k for k,v in mapping.items()})[list(mapping)].copy()
    for c in ['contract_date','available_date']:
        d[c]=pd.to_datetime(d[c],errors='raise').dt.normalize()
    d['price']=pd.to_numeric(d.price,errors='raise')
    if d[['contract_date','available_date','price','gnaf_id']].isna().any().any(): raise ValueError('Missing required training values')
    if not np.isfinite(d.price.to_numpy(dtype=float)).all() or (d.price<=0).any():
        raise ValueError('Training prices must be finite and positive')
    cut=pd.Timestamp(cutoff)
    if cut>pd.Timestamp('2026-08-31'): raise ValueError('Official information cutoff exceeded')
    # Refuse contamination, rather than silently training on competition target sales.
    if (d.contract_date>cut).any(): raise ValueError('Training includes contracts after valuation cutoff')
    before=len(d); d=d[(d.available_date<=cut)&(d.available_date>=d.contract_date)].copy()
    cols=['gnaf_id','unit','house_number','street','suburb','postcode','building_key','property_key']
    d=d.merge(targets[cols],on='gnaf_id',how='inner',validate='many_to_one')
    # The optional model view keeps the earliest whole row, including missing
    # attributes. The source fact table is maintained separately by its caller.
    if type(deduplicate) is not bool:
        raise ValueError('deduplicate must be boolean')
    d=d.sort_values('available_date',kind='stable')
    linked_rows=len(d)
    if deduplicate:
        d=d.drop_duplicates(['gnaf_id','contract_date','price'])
    if not len(d): raise ValueError('No eligible, linked training records')
    if 'area_sqm' not in d: d['area_sqm']=np.nan
    d['area_sqm']=pd.to_numeric(d.area_sqm,errors='coerce')
    d['row_id']=['official_'+str(i) for i in range(len(d))]
    return d,{'raw_n':before,'eligible_linked_n':len(d),'cutoff':cutoff,
              'deduplicate':deduplicate,'duplicate_events_removed':linked_rows-len(d),
              'availability_source_column':mapping['available_date'],
              'note':'Time fields follow the confirmed source-time protocol, including event time when explicitly authorized; publication time is not inferred.'}

def validate_submission(input_ids,output,expected_rows=181711):
    ids=pd.Series(input_ids,dtype=str).reset_index(drop=True)
    if set(output.columns)!={'gnaf_id','value'}: raise ValueError('Output columns must be gnaf_id,value')
    if expected_rows is not None and len(ids)!=expected_rows: raise ValueError('Incorrect input row count')
    if ids.isna().any() or ids.str.strip().eq('').any() or ids.duplicated().any(): raise ValueError('Invalid input IDs')
    actual=output.gnaf_id.astype(str)
    if len(output)!=len(ids) or actual.duplicated().any() or set(actual)!=set(ids): raise ValueError('Output IDs must match input exactly')
    values=pd.to_numeric(output.value,errors='coerce').to_numpy(float)
    if not np.isfinite(values).all() or (values<=0).any() or not (values==np.floor(values)).all():
        raise ValueError('Prices must be finite positive whole dollars')
    return {'n':len(output),'coverage':1.,'min':int(values.min()),'max':int(values.max())}

def write_submission(ids,predictions,path,expected_rows=181711):
    p=np.asarray(predictions,dtype=float)
    if not np.isfinite(p).all() or (p<=0).any(): raise ValueError('Invalid model predictions')
    out=pd.DataFrame({'gnaf_id':list(ids),'value':np.rint(p).astype('int64')})
    result=validate_submission(ids,out,expected_rows)
    path=Path(path)
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True); out.to_csv(path,index=False)
    return result
