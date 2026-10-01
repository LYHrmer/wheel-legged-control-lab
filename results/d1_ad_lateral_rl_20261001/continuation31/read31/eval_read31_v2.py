"""C31 final evaluation v2: same physical and gain gates, corrected source seam."""
from types import FunctionType

import eval_read31 as frozen
from source_read31_v2 import sources31 as verified_sources31


def sources31(run, session, worker, *, reader_review_path=None):
    # The original numerical orchestrator passes its original review filename.
    # This new entry is independently bound to the new v2 review explicitly.
    del reader_review_path
    return verified_sources31(run, session, worker,
        reader_review_path=frozen.P/'eval_reader_source_review_31_v2.json')


_namespace = dict(frozen.read_evaluation31.__globals__, sources31=sources31)
read_evaluation31 = FunctionType(frozen.read_evaluation31.__code__, _namespace,
                                'read_evaluation31_v2')
_entry_namespace = dict(frozen.main.__globals__, read_evaluation31=read_evaluation31)
main = FunctionType(frozen.main.__code__, _entry_namespace, 'main_v2')


if __name__ == '__main__':
    main()
