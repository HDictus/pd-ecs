import pd_ecs


def test_subcomponents_are_tuples():
    x = pd_ecs.Component('x')
    y = pd_ecs.Component('y')
    posn = pd_ecs.Component(x=x, y=y, name='position')
    assert posn.x == (posn, x)
    assert posn.y == (posn, y)