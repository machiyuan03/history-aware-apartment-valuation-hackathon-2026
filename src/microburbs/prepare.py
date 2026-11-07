"""Address normalization used by the final valuation feature pipeline."""
import re

STREETS = {'STREET':'ST','ROAD':'RD','AVENUE':'AVE','PARADE':'PDE','PLACE':'PL',
           'LANE':'LANE','DRIVE':'DR','CRESCENT':'CRES','TERRACE':'TCE','SQUARE':'SQ',
           'COURT':'CT','BOULEVARD':'BVD','HIGHWAY':'HWY','CIRCUIT':'CCT'}

def norm(value):
    return re.sub(r'\s+', ' ', str(value).upper().strip()).replace(' (NSW)', '')

def norm_street(value):
    return ' '.join(STREETS.get(w, w) for w in norm(value).replace('.', '').split())

def address_keys(frame):
    d=frame.copy()
    for c in ['unit','house_number','street','suburb','postcode']:
        d[c]=d[c].fillna('').map(norm_street if c=='street' else norm)
    # Whitespace inside a number does not define a different address: '2 A' == '2A'.
    for c in ['unit','house_number']:
        d[c]=d[c].str.replace(' ', '', regex=False)
    d['building_key']=d['house_number']+'|'+d['street']+'|'+d['suburb']+'|'+d['postcode']
    d['property_key']=d['unit']+'|'+d['building_key']
    return d

