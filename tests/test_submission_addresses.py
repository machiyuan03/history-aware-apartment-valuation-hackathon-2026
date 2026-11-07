import numpy as np
import pandas as pd
import pytest

from microburbs.submission import adapt_input, adapt_training


def inputs(addresses, units=None):
    return pd.DataFrame({'gnaf_id': [f'id{i}' for i in range(len(addresses))],
                         'address': addresses, 'unit_number': units or [''] * len(addresses),
                         'suburb': 'Sydney', 'postcode': '2000'})


def test_levels_preserve_subpremise_identity_and_share_only_the_building():
    raw = inputs(['Suite 1, Level 1, 822 George Street',
                  'Suite 1, Level 2, 822 George Street',
                  'Level 1, 822 George Street', 'Level 2, 822 George Street',
                  'Shop 1, 822 George Street', 'Unit 1, 822 George Street'])
    result, quality = adapt_input(raw, expected_rows=None)
    assert quality['address_parse_failures'] == 0
    assert result.house_number.eq('822').all() and result.street.eq('GEORGE ST').all()
    assert result.building_key.nunique() == 1
    assert result.property_key.nunique() == len(raw)
    assert result.address_level.tolist()[:4] == ['LEVEL 1', 'LEVEL 2', 'LEVEL 1', 'LEVEL 2']
    assert result.unit.tolist()[2:4] == ['', '']  # Floors are not invented unit numbers.
    pd.testing.assert_series_equal(result.address, raw.address)
    pd.testing.assert_series_equal(result.unit_number, raw.unit_number)


@pytest.mark.parametrize('prefix', ['Carspace 12', 'Office 3B', 'Store 4', 'Kiosk 2',
                                    'Lower Ground Floor QG', 'Basement 1', 'Mezzanine 2'])
def test_explicit_nonresidential_components_are_retained(prefix):
    raw = inputs([prefix + ', 15 Bourke Road'])
    result, quality = adapt_input(raw, expected_rows=None)
    assert quality['address_parse_failures'] == 0 and len(result) == 1
    assert result.house_number.iloc[0] == '15' and result.street.iloc[0] == 'BOURKE RD'
    assert result.address.iloc[0] == raw.address.iloc[0]
    assert result.address_unit_type.iloc[0] or result.address_level.iloc[0]


def test_distinct_ids_with_equal_visible_addresses_never_merge():
    raw = inputs(['Shop 1, 43 Wyndham Street', 'Shop 1, 43 Wyndham Street',
                  '10 King Street', '10 King Street'])
    result, quality = adapt_input(raw, expected_rows=None)
    assert result.property_key.nunique() == len(raw)
    assert result.address_property_key.nunique() == 2  # Optional address lookup view retained.
    assert quality['address_key_collision_rows'] == 4
    assert quality['address_key_collision_groups'] == 2
    reordered, _ = adapt_input(raw.iloc[::-1], expected_rows=None)
    assert result.set_index('gnaf_id').property_key.to_dict() == reordered.set_index('gnaf_id').property_key.to_dict()


def test_unknown_prefix_and_unit_conflict_cannot_be_falsely_matched():
    raw = inputs(['Unknown wing, 10 King Street', 'Unknown wing, 10 King Street',
                  'Unit 2, 10 King Street', 'Unit 1, 10 King Street'], ['', '', '1', '1'])
    result, quality = adapt_input(raw, expected_rows=None)
    assert quality['address_parse_failures'] == 2
    assert quality['unit_number_conflicts'] == 1
    assert result.property_key.nunique() == len(raw)
    assert result.building_key.iloc[0] != result.building_key.iloc[1]
    assert result.unit_number_conflict.tolist() == [False, False, True, False]


def test_training_retains_positive_extreme_prices_without_global_clipping():
    target, _ = adapt_input(inputs(['Unit 1, 10 King Street']), expected_rows=None)
    raw = pd.DataFrame({'gnaf_id': ['id0', 'id0'], 'price': [1, 30000000],
                        'contract_date': ['2020-01-01', '2021-01-01'],
                        'available_date': ['2020-01-01', '2021-01-01']})
    adapted, quality = adapt_training(raw, target, {c: c for c in raw})
    assert adapted.price.tolist() == [1, 30000000]
    assert quality['eligible_linked_n'] == 2


def test_duplicate_event_view_can_retain_all_linked_source_rows():
    target, _ = adapt_input(inputs(['Unit 1, 10 King Street']), expected_rows=None)
    raw = pd.DataFrame({'gnaf_id': ['id0', 'id0'], 'price': [1, 1],
                        'contract_date': ['2020-01-01', '2020-01-01'],
                        'available_date': ['2020-01-01', '2020-02-01']})
    adapted, quality = adapt_training(raw, target, {c: c for c in raw}, deduplicate=False)
    assert len(adapted) == 2 and quality['duplicate_events_removed'] == 0
    assert raw.available_date.tolist() == ['2020-01-01', '2020-02-01']


@pytest.mark.parametrize('price', [0, -1, np.nan, np.inf, -np.inf])
def test_training_rejects_nonpositive_or_nonfinite_prices(price):
    target, _ = adapt_input(inputs(['Unit 1, 10 King Street']), expected_rows=None)
    raw = pd.DataFrame({'gnaf_id': ['id0'], 'price': [price],
                        'contract_date': ['2020-01-01'], 'available_date': ['2020-01-01']})
    with pytest.raises(ValueError):
        adapt_training(raw, target, {c: c for c in raw})
