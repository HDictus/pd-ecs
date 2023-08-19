import pytest
from pd_ecs.component import Component


def test_components_have_fields_as_attributes():
    position = Component(x='x', y='y', has_space='has a space',
                          name='component')
    assert position.x == (position, 'x')
    assert position.y == (position, 'y')
    assert position.fields == ['x', 'y', 'has a space']

    position = Component('x', 'y', 'A space')
    assert position.x == (position, 'x')
    assert position.y == (position, 'y')
    assert getattr(position, 'A space') == (position, 'A space')
    assert position.fields == ['x', 'y', 'A space']

    with pytest.raises(ValueError):
        position = Component('init_dataframe')
