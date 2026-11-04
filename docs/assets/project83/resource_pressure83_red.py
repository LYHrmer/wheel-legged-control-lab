"""Additional pressure policy for new83 jobs; old guard remains frozen."""
from scripts.resource_guard42 import check_memory


def check_pressure(snapshot, pressures, *, startup=False):
    # Initial reproduction: the previous acceptance criterion ignores PSI/high.
    return check_memory(snapshot)
