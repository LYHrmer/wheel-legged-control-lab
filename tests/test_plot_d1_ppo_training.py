"""Offline contract tests; no project, Torch, or simulation imports."""
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/plot_d1_ppo_training.py'
METRICS = ('train/std', 'train/approx_kl', 'train/clip_fraction', 'train/value_loss',
           'train/policy_gradient_loss', 'train/explained_variance')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n', encoding='utf-8')


def fixture(study):
    protocol = {
        'schema': 'd1-budget-study-v1', 'kind': 'smoke', 'workers': 4, 'n_steps': 128,
        'training_seeds': [31001], 'action_modes': ['shared2', 'independent8'],
        'checkpoint_budgets': [512, 1024], 'expected_train_calls': 4,
        'continuous_training_trajectories': 2, 'checkpoint_count': 4,
        'ppo_settings': {'n_steps': 128, 'n_epochs': 4}, 'models': [], 'checkpoints': [],
    }
    for mode in protocol['action_modes']:
        name = f'{mode}_seed31001'
        protocol['models'].append({'model_id': name, 'training_run': name,
                                   'action_mode': mode, 'seed': 31001})
        model = study / name
        dump(model / 'protocol.json', {
            'schema': 'd1-command-locomotion-experiment-v1',
            'arguments': {'seed': 31001, 'action_mode': mode, 'mode': 'train',
                          'workers': 4, 'steps': 1024},
            'checkpoint_budgets': [512, 1024], 'ppo_settings': {'n_steps': 128, 'n_epochs': 4},
            'action_contract': {'action_mode': mode},
        })
        checkpoint_bytes = f'fixture checkpoint for {name}'.encode()
        checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
        dump(model / 'training.json', {
            'training_mode': 'single_continuous_learn', 'learn_calls': 1,
            'num_timesteps': 1024, 'updates': 8, 'checkpoint_budgets': [512, 1024],
            'training_finished_at_utc': '2026-09-11T00:00:00+00:00',
            'action_contract': {'action_mode': mode}, 'checkpoint_sha256': checkpoint_hash,
        })
        for index, budget in enumerate(protocol['checkpoint_budgets']):
            prefix = f'{name}/checkpoints/budget{budget}'
            metadata = f'{prefix}/checkpoint.json'
            zip_path = f'{prefix}/checkpoint.zip'
            protocol['checkpoints'].append({
                'model_id': name, 'training_run': name, 'budget': budget,
                'expected_update_index': index, 'checkpoint_id': f'{name}_budget{budget}',
                'relative_metadata': metadata, 'relative_zip': zip_path,
            })
            dump(study / metadata, {'schema': 'd1-locomotion-checkpoint-v1',
                'action_mode': mode, 'model_sha256': checkpoint_hash,
                'extra': {'training_seed': 31001, 'checkpoint_budget': budget,
                          'num_timesteps': budget, 'after_update_index': index}})
            (study / zip_path).write_bytes(checkpoint_bytes)
        rows = []
        for index in range(2):
            rows.append({'audit_index': index, 'num_timesteps': 512 * (index + 1),
                         'n_updates': 4 * (index + 1),
                         'logger_train_metrics': dict(zip(METRICS,
                             [0.135, 0.02 + 0.01 * index, 0.1, 0.02, -0.008, -0.08])),
                         'approx_reverse_kl_after': 99999,
                         'explained_variance_old_values': 88888})
        log = model / 'updates/updates.jsonl'
        log.parent.mkdir()
        log.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    dump(study / 'protocol.json', protocol)


class TrainingDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('plot_d1_ppo_training_test', SCRIPT)
        cls.plotter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.plotter)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='d1-ppo-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study = self.root / 'study'
        self.output = self.root / 'plots'
        fixture(self.study)
        self.log = self.study / 'shared2_seed31001/updates/updates.jsonl'

    def run_cli(self):
        return subprocess.run([sys.executable, str(SCRIPT), '--study', str(self.study),
                               '--output', str(self.output)], capture_output=True,
                              text=True, timeout=30, check=False)

    def assert_rejected(self, match=None):
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertNotIn('Traceback', result.stderr)
        if match:
            self.assertIn(match, result.stderr)
        self.assertFalse(self.output.exists())

    def rewrite_rows(self, mutate):
        rows = [json.loads(line) for line in self.log.read_text().splitlines()]
        mutate(rows)
        self.log.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def test_complete_smoke_csv_and_manifest_hashes(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.output / 'training_metrics.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 24)
        self.assertEqual({row['metric'] for row in rows}, set(METRICS))
        self.assertEqual({row['num_timesteps'] for row in rows}, {'512', '1024'})
        self.assertTrue(all(float(row['value']) < 1 for row in rows))
        manifest = json.loads((self.output / 'manifest.json').read_text())
        self.assertEqual(manifest['kind'], 'smoke')
        self.assertTrue(set(METRICS).issubset(manifest['metrics_semantics']))
        self.assertTrue((self.output / 'training_diagnostics.png').stat().st_size > 10000)
        self.assertEqual(manifest['script']['sha256'], hashlib.sha256(SCRIPT.read_bytes()).hexdigest())
        for name, expected in manifest['inputs_sha256'].items():
            self.assertEqual(hashlib.sha256((self.study / name).read_bytes()).hexdigest(), expected)
        for name, expected in manifest['outputs_sha256'].items():
            self.assertEqual(hashlib.sha256((self.output / name).read_bytes()).hexdigest(), expected)

    def test_missing_study(self):
        self.study = self.root / 'absent'
        self.assert_rejected()

    def test_output_exists_preserved(self):
        self.output.mkdir()
        sentinel = self.output / 'existing.txt'
        sentinel.write_text('preserve this')
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_text(), 'preserve this')
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_output_nested_in_study(self):
        self.output = self.study / 'plots'
        self.assert_rejected()

    def test_missing_training_marker(self):
        (self.study / 'shared2_seed31001/training.json').unlink()
        self.assert_rejected()

    def test_missing_checkpoint(self):
        (self.study / 'shared2_seed31001/checkpoints/budget512/checkpoint.zip').unlink()
        self.assert_rejected()

    def test_checkpoint_hash_mismatch(self):
        (self.study / 'shared2_seed31001/checkpoints/budget512/checkpoint.zip').write_bytes(b'changed')
        self.assert_rejected()

    def test_missing_log(self):
        self.log.unlink()
        self.assert_rejected()

    def test_missing_row(self):
        self.rewrite_rows(lambda rows: rows.pop())
        self.assert_rejected()

    def test_duplicate_row(self):
        self.rewrite_rows(lambda rows: rows.__setitem__(1, rows[0]))
        self.assert_rejected()

    def test_out_of_sequence_rows(self):
        self.rewrite_rows(lambda rows: rows.reverse())
        self.assert_rejected()

    def test_nonmonotonic_timesteps(self):
        self.rewrite_rows(lambda rows: rows[1].__setitem__('num_timesteps', 512))
        self.assert_rejected()

    def test_missing_metric(self):
        self.rewrite_rows(lambda rows: rows[0]['logger_train_metrics'].pop('train/std'))
        self.assert_rejected()

    def test_nonfinite_metric(self):
        self.rewrite_rows(lambda rows: rows[0]['logger_train_metrics'].__setitem__('train/std', float('nan')))
        self.assert_rejected()

    def test_boolean_metric(self):
        self.rewrite_rows(lambda rows: rows[0]['logger_train_metrics'].__setitem__('train/std', True))
        self.assert_rejected()

    def test_boolean_index(self):
        self.rewrite_rows(lambda rows: rows[0].__setitem__('audit_index', False))
        self.assert_rejected()

    def test_malformed_json_line(self):
        self.log.write_text(self.log.read_text() + '{broken\n')
        self.assert_rejected()

    def test_duplicate_json_key(self):
        self.log.write_text(self.log.read_text().replace('"audit_index": 0', '"audit_index": 0, "audit_index": 0', 1))
        self.assert_rejected()

    def test_blank_line(self):
        self.log.write_text(self.log.read_text() + '\n')
        self.assert_rejected()

    def test_nonobject_protocol(self):
        dump(self.study / 'protocol.json', [])
        self.assert_rejected()

    def test_boolean_workers(self):
        path = self.study / 'protocol.json'
        protocol = json.loads(path.read_text())
        protocol['workers'] = True
        dump(path, protocol)
        self.assert_rejected()

    def test_fractional_index(self):
        self.rewrite_rows(lambda rows: rows[0].__setitem__('audit_index', 0.0))
        self.assert_rejected()

    def test_checkpoint_identity(self):
        path = self.study / 'protocol.json'
        protocol = json.loads(path.read_text())
        protocol['checkpoints'][0]['relative_zip'] = protocol['checkpoints'][1]['relative_zip']
        dump(path, protocol)
        self.assert_rejected()

    def test_fractional_checkpoint_metadata(self):
        path = self.study / 'shared2_seed31001/checkpoints/budget512/checkpoint.json'
        meta = json.loads(path.read_text())
        meta['extra']['num_timesteps'] = 512.5
        dump(path, meta)
        self.assert_rejected()

    def test_root_final_zip_hash_need_not_match_budget_zip(self):
        path = self.study / 'shared2_seed31001/training.json'
        training = json.loads(path.read_text())
        training['checkpoint_sha256'] = 'a' * 64
        dump(path, training)
        inputs = self.plotter.Inputs(self.study)
        spec = self.plotter.load_protocol(inputs)
        series = [self.plotter.load_model(inputs, spec, model) for model in spec['models']]
        self.assertEqual(len(series), 2)

    def test_duplicate_model(self):
        path = self.study / 'protocol.json'
        protocol = json.loads(path.read_text())
        protocol['models'][1] = protocol['models'][0]
        dump(path, protocol)
        self.assert_rejected()

    def test_incomplete_training(self):
        path = self.study / 'shared2_seed31001/training.json'
        training = json.loads(path.read_text())
        training['num_timesteps'] = 512
        dump(path, training)
        self.assert_rejected()

    def test_mutated_input_refuses_publish_and_cleans_staging(self):
        original = self.plotter.render

        def render_and_mutate(*args):
            original(*args)
            self.log.write_text(self.log.read_text() + '\n')

        with patch.object(self.plotter, 'render', side_effect=render_and_mutate):
            code = self.plotter.main(['--study', str(self.study), '--output', str(self.output)])
        self.assertEqual(code, 2)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('.plots.staging.*')))

    def test_figure_preserves_all_raw_series_and_smoke_label(self):
        inputs = self.plotter.Inputs(self.study)
        spec = self.plotter.load_protocol(inputs)
        series = [self.plotter.load_model(inputs, spec, model) for model in spec['models']]
        self.output.mkdir()
        with patch.object(self.plotter.plt, 'close'):
            self.plotter.render(self.output, spec, series)
            figure = self.plotter.plt.gcf()
        try:
            self.assertEqual(len(figure.axes), 6)
            self.assertTrue(any('SMOKE - PIPELINE CHECK ONLY' in t.get_text() for t in figure.texts))
            for axis, metric in zip(figure.axes, METRICS):
                self.assertEqual(axis.get_title(), metric)
                self.assertEqual(len(axis.lines), 2)
                self.assertEqual(axis.get_xscale(), 'linear')
                self.assertEqual(axis.get_yscale(), 'linear')
                for line, entry in zip(axis.lines, series):
                    self.assertEqual(list(line.get_xdata()), [512, 1024])
                    self.assertEqual(list(line.get_ydata()), [r['metrics'][metric] for r in entry['rows']])
        finally:
            self.plotter.plt.close(figure)


if __name__ == '__main__':
    unittest.main()
