"""C31 training readback v2: source readiness fix; frozen numerical reader reused."""
from types import FunctionType

import read31 as frozen
from source_read31_v2 import sources31

# A private function namespace changes only the reviewed source-check seam.
# The original module, function and numerical helper globals are never mutated.
_namespace = dict(frozen.read_training31.__globals__, sources31=sources31)
read_training31 = FunctionType(frozen.read_training31.__code__, _namespace,
                              'read_training31_v2')
_entry_namespace = dict(frozen.main.__globals__, read_training31=read_training31)
main = FunctionType(frozen.main.__code__, _entry_namespace, 'main_v2')


if __name__ == '__main__':
    main()
