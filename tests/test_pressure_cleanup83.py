"""Mock process ownership and pidfds; never signal/read a real cgroup in CI."""

from __future__ import annotations

import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import run_pressure83 as runner

SCOPE = "/user.slice/probe83.service"


class PressureCleanup83(unittest.TestCase):
    def test_self_is_rejected_before_pidfd_open(self):
        with (
            mock.patch.object(runner.os, "getpid", return_value=111),
            mock.patch.object(runner.os, "pidfd_open") as opened,
        ):
            with self.assertRaisesRegex(ValueError, "supervisor"):
                runner.signal_owned_pid(SCOPE, 111, signal.SIGTERM)
            opened.assert_not_called()

    def test_outside_or_ambiguous_cgroup_is_rejected_and_pidfd_closed(self):
        for contents in (
            "0::/user.slice/other.service",
            "0::/user.slice/probe83.service_extra",
            "0::" + SCOPE + "\n0::" + SCOPE,
        ):
            with (
                self.subTest(contents=contents),
                mock.patch.object(runner.os, "getpid", return_value=111),
                mock.patch.object(runner.os, "pidfd_open", return_value=77),
                mock.patch.object(runner, "_read", return_value=contents),
                mock.patch.object(runner.signal, "pidfd_send_signal") as sent,
                mock.patch.object(runner.os, "close") as closed,
            ):
                with self.assertRaisesRegex(RuntimeError, "outside owned cgroup"):
                    runner.signal_owned_pid(SCOPE, 222, signal.SIGTERM)
                sent.assert_not_called()
                closed.assert_called_once_with(77)

    def test_verified_descendant_uses_pidfd_and_closes_it(self):
        for scope in (SCOPE, SCOPE + "/nested"):
            with (
                self.subTest(scope=scope),
                mock.patch.object(runner.os, "getpid", return_value=111),
                mock.patch.object(runner.os, "pidfd_open", return_value=77) as opened,
                mock.patch.object(runner, "_read", return_value="0::" + scope),
                mock.patch.object(runner.signal, "pidfd_send_signal") as sent,
                mock.patch.object(runner.os, "close") as closed,
            ):
                self.assertTrue(runner.signal_owned_pid(SCOPE, 222, signal.SIGKILL))
                opened.assert_called_once_with(222, 0)
                sent.assert_called_once_with(77, signal.SIGKILL, None, 0)
                closed.assert_called_once_with(77)

    def test_exited_process_paths_do_not_leak_pidfd(self):
        with (
            mock.patch.object(runner.os, "getpid", return_value=111),
            mock.patch.object(runner.os, "pidfd_open", side_effect=ProcessLookupError),
            mock.patch.object(runner.os, "close") as closed,
        ):
            self.assertFalse(runner.signal_owned_pid(SCOPE, 222, signal.SIGTERM))
            closed.assert_not_called()
        for read_error, send_error in ((FileNotFoundError, None), (None, ProcessLookupError)):
            with (
                self.subTest(read_error=read_error, send_error=send_error),
                mock.patch.object(runner.os, "getpid", return_value=111),
                mock.patch.object(runner.os, "pidfd_open", return_value=77),
                mock.patch.object(
                    runner, "_read", side_effect=read_error, return_value="0::" + SCOPE
                ),
                mock.patch.object(runner.signal, "pidfd_send_signal", side_effect=send_error),
                mock.patch.object(runner.os, "close") as closed,
            ):
                self.assertFalse(runner.signal_owned_pid(SCOPE, 222, signal.SIGTERM))
                closed.assert_called_once_with(77)

    def test_members_include_nested_cgroups_but_exclude_self(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            owned = root / SCOPE[1:]
            child = owned / "nested"
            child.mkdir(parents=True)
            (owned / "cgroup.procs").write_text("111\n222\n")
            (child / "cgroup.procs").write_text("333\n")
            with (
                mock.patch.object(runner, "_self_scope", return_value=SCOPE),
                mock.patch.object(runner.os, "getpid", return_value=111),
            ):
                self.assertEqual(runner.owned_members(SCOPE, root), {222, 333})

    def test_changed_inode_or_supervisor_scope_rejects_enumeration(self):
        directory = mock.MagicMock(spec=Path)
        directory.__truediv__.return_value = Path("/mock-cgroup") / SCOPE[1:] / "cgroup.procs"
        directory.glob.return_value = []
        directory.stat.side_effect = [
            SimpleNamespace(st_dev=1, st_ino=100),
            SimpleNamespace(st_dev=1, st_ino=101),
        ]
        with (
            mock.patch.object(runner, "_self_scope", return_value=SCOPE),
            mock.patch.object(runner, "_directory", return_value=directory),
            mock.patch.object(runner, "_read", return_value="222"),
            self.assertRaisesRegex(RuntimeError, "identity changed"),
        ):
            runner.owned_members(SCOPE, Path("/mock-cgroup"))
        with (
            mock.patch.object(runner, "_self_scope", return_value="/elsewhere"),
            mock.patch.object(runner, "_directory") as accessed,
        ):
            with self.assertRaisesRegex(RuntimeError, "left its registered cgroup"):
                runner.owned_members(SCOPE)
            accessed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
