import unittest
from unittest import mock

from host import live_fpga


class LiveGuardDeploymentTest(unittest.TestCase):
    @mock.patch('host.live_fpga.subprocess.run')
    def test_unique_deployment_and_guard(self, run):
        first = live_fpga.push_board_files('board', 'root', 'key', 22)
        second = live_fpga.push_board_files('board', 'root', 'key', 22)
        self.assertNotEqual(first, second)
        self.assertTrue(first[0].endswith('/fpga_live_runner.py'))
        self.assertTrue(first[1].endswith('/temp_guard.sh'))
        self.assertEqual(run.call_count, 6)
        client = live_fpga.LiveFpgaClient(runner_remote=first[0], monitor_remote=first[1])
        cmd = client._remote_cmd()
        self.assertIn(first[1], cmd)
        self.assertIn('wait $G', cmd)
        self.assertIn('75 1', cmd)

    @mock.patch('host.live_preflight.run_probe', return_value={'reachable': '0'})
    @mock.patch.object(live_fpga.LiveFpgaClient, 'check_connection', return_value=True)
    @mock.patch('host.live_fpga.subprocess.Popen')
    def test_preflight_failure_prevents_inference(self, popen, connection, probe):
        events = list(live_fpga.LiveFpgaClient().stream())
        self.assertIn('preflight failed', events[0]['msg'])
        popen.assert_not_called()

    @mock.patch('host.live_preflight.evaluate', return_value=[{'check': 'ok', 'ok': True, 'detail': 'ok'}])
    @mock.patch('host.live_preflight.run_probe', return_value={})
    @mock.patch.object(live_fpga.LiveFpgaClient, 'check_connection', return_value=True)
    @mock.patch('host.live_fpga.subprocess.Popen')
    def test_missing_guard_prevents_inference(self, popen, connection, probe, evaluate):
        events = list(live_fpga.LiveFpgaClient().stream())
        self.assertIn('no independent temperature guard', events[0]['msg'])
        popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
