from pd_ecs import System
from mock import MagicMock

def test_system_initialized_adds_to_world():
    mockworld = MagicMock()
    sys = System(mockworld)
    assert mockworld.add_system.called_with(sys)
    assert sys.world == mockworld
