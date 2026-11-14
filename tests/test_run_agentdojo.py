import argparse
import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agentdojo.attacks.attack_registry import ATTACKS
from agentdojo.scripts import benchmark
from my_attacks import FromMultiTurnSearchAttack, FromSearchAttack
from run_agentdojo import (
    configured_attack_loader,
    main,
    parse_attack_kwargs,
    result_path,
    run_case,
    save_run_config,
    validate_attack_kwargs,
)


class ParameterAttack:
    def __init__(self, suite, pipeline, *, enabled=False, options=None):
        self.suite, self.pipeline = suite, pipeline
        self.enabled, self.options = enabled, options


class AttackKwargsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = SimpleNamespace(
            model_id="test-model", logdir=self.root, attack="parameter_test",
            attack_kwargs={"enabled": True}, modules_to_load=["my_attacks"], recap=False,
        )
        registration = patch.dict(ATTACKS, parameter_test=ParameterAttack)
        registration.start()
        self.addCleanup(registration.stop)

    def test_parses_json_types_and_rejects_non_objects_or_nonfinite_numbers(self):
        kwargs = parse_attack_kwargs('{"enabled":true,"options":{"n":2,"values":[1,null,"x"]}}')
        self.assertIs(kwargs["enabled"], True)
        self.assertEqual(kwargs["options"], {"n": 2, "values": [1, None, "x"]})
        for raw in ("broken", "[]", "null", '"text"', "1", '{"n":NaN}', '{"n":1e400}'):
            with self.subTest(raw=raw), self.assertRaises(argparse.ArgumentTypeError):
                parse_attack_kwargs(raw)

    def test_validates_constructor_without_instantiating(self):
        validate_attack_kwargs("parameter_test", {"enabled": True, "options": {"n": 2}})
        validate_attack_kwargs("dummy", {})
        validate_attack_kwargs(None, {})
        for name, kwargs in (("parameter_test", {"unknown": 1}),
                             ("parameter_test", {"suite": "overridden"}),
                             ("not_registered", {}), (None, {"enabled": True})):
            with self.subTest(name=name, kwargs=kwargs), self.assertRaises(ValueError):
                validate_attack_kwargs(name, kwargs)

    def test_scoped_loader_forwards_typed_values_and_preserves_other_attacks(self):
        suite, pipeline = object(), object()
        kwargs = {"enabled": True, "options": {"n": 2, "values": [None, False]}}
        original = Mock(return_value="other attack")
        with patch.object(benchmark, "load_attack", original):
            with configured_attack_loader("parameter_test", kwargs):
                attack = benchmark.load_attack("parameter_test", suite, pipeline)
                self.assertIs(attack.suite, suite)
                self.assertIs(attack.pipeline, pipeline)
                self.assertIs(attack.enabled, True)
                self.assertEqual(attack.options, kwargs["options"])
                self.assertEqual(benchmark.load_attack("dummy", suite, pipeline), "other attack")
            self.assertIs(benchmark.load_attack, original)
        original.assert_called_once_with("dummy", suite, pipeline)
        self.assertIs(ATTACKS["parameter_test"], ParameterAttack)

    def test_loader_restores_on_system_exit_and_is_unchanged_for_empty_kwargs(self):
        original = benchmark.load_attack
        with configured_attack_loader("parameter_test", {}):
            self.assertIs(benchmark.load_attack, original)
        with self.assertRaises(SystemExit):
            with configured_attack_loader("parameter_test", {"enabled": True}):
                raise SystemExit(0)
        self.assertIs(benchmark.load_attack, original)

    def test_configuration_round_trips_and_allows_worker_or_suite_changes(self):
        save_run_config(self.args)
        saved = json.loads((self.root / "run_config.json").read_text())
        self.assertEqual(saved["attack_kwargs"], {"enabled": True})
        self.args.jobs, self.args.suites = 32, ["banking"]
        save_run_config(self.args)
        self.assertEqual(json.loads((self.root / "run_config.json").read_text()), saved)

    def test_configuration_rejects_different_parameters_and_keeps_original(self):
        save_run_config(self.args)
        path = self.root / "run_config.json"
        original = path.read_text()
        changes = {
            "attack_kwargs": {"enabled": 1}, "model_id": "another-model",
            "attack": "dummy", "recap": True, "modules_to_load": [],
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                changed = copy.deepcopy(self.args)
                setattr(changed, key, value)
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    save_run_config(changed)
                self.assertEqual(path.read_text(), original)

    def test_legacy_completed_results_require_empty_kwargs(self):
        path = result_path(self.root, ("banking", "user_task_1", "injection_task_0"), self.args.attack)
        path.parent.mkdir(parents=True)
        path.write_text('{"utility":true}')
        with self.assertRaisesRegex(ValueError, "no run_config.json"):
            save_run_config(self.args)
        self.assertFalse((self.root / "run_config.json").exists())
        self.args.attack_kwargs = {}
        save_run_config(self.args)
        self.assertTrue((self.root / "run_config.json").is_file())

    def test_main_rejects_bad_constructor_kwargs_before_workers_or_config(self):
        error = io.StringIO()
        with (
            patch("run_agentdojo.setup_agentdojo"),
            patch("run_agentdojo.load_endpoint") as endpoint,
            patch("run_agentdojo.mp.get_context") as context,
            redirect_stderr(error),
            self.assertRaises(SystemExit) as raised,
        ):
            main(["--model-id", "test-model", "--logdir", str(self.root),
                  "--attack", "parameter_test", "--attack-kwargs", '{"typo":true}'])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unexpected keyword argument 'typo'", error.getvalue())
        endpoint.assert_not_called()
        context.assert_not_called()
        self.assertFalse((self.root / "run_config.json").exists())

    def test_main_records_cli_kwargs_without_starting_completed_cases(self):
        with (
            patch("run_agentdojo.setup_agentdojo"),
            patch("run_agentdojo.load_endpoint"),
            patch("run_agentdojo.list_cases", return_value=[]),
            patch("run_agentdojo.mp.get_context") as context,
            redirect_stdout(io.StringIO()),
        ):
            main(["--model-id", "test-model", "--logdir", str(self.root),
                  "--attack", "parameter_test", "--attack-kwargs", '{"enabled":true}',
                  "-ml", "my_attacks"])
        context.return_value.Process.assert_not_called()
        saved = json.loads((self.root / "run_config.json").read_text())
        self.assertEqual(saved["attack_kwargs"], {"enabled": True})
        self.assertEqual(saved["modules_to_load"], ["my_attacks"])


class SearchAttackParameterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.template = self.root / "模板 template.json"
        self.template.write_text(json.dumps({"prefix": '前缀 "quoted"\n', "suffix": "\n后缀 {literal}"}))
        self.payload = self.root / "payload.json"
        self.payload.write_text(json.dumps({"messages": [
            {"role": "user", "stage": "C", "content": "{goal}"},
        ]}))
        self.suite = Mock()
        self.suite.get_injection_vector_defaults.return_value = {"injection": ""}

    def test_from_search_uses_constructor_or_default_and_ignores_environment(self):
        alternate = self.root / "alternate.json"
        alternate.write_text('{"prefix":"alternate ","suffix":"!"}')
        with (
            patch.dict(os.environ, FROM_SEARCH_TEMPLATE="missing-file.json"),
            patch("my_attacks.FROM_SEARCH_TEMPLATE", self.template),
        ):
            default = FromSearchAttack(self.suite, None)
            selected = FromSearchAttack(self.suite, None, template_path=str(alternate))
        self.assertEqual(default.template_path, self.template)
        self.assertEqual(default.render("GOAL {goal}"), '前缀 "quoted"\nGOAL {goal}\n后缀 {literal}')
        with patch.object(selected, "get_injection_candidates", return_value=["injection"]):
            self.assertEqual(selected.attack(None, SimpleNamespace(GOAL="GOAL")),
                             {"injection": "alternate GOAL!"})

    def test_case_routes_both_search_attack_paths_through_the_benchmark_loader(self):
        original = benchmark.load_attack
        cases = (
            ("from_search", "template_path", self.template, FromSearchAttack),
            ("from_multi_turn_search", "payload_path", self.payload, FromMultiTurnSearchAttack),
        )
        for name, parameter, path, attack_class in cases:
            with self.subTest(attack=name):
                args = SimpleNamespace(
                    model_id="test-model", logdir=self.root, attack=name,
                    attack_kwargs={parameter: str(path)}, modules_to_load=["my_attacks"], recap=False,
                )
                validate_attack_kwargs(name, args.attack_kwargs)

                def benchmark_main(argv):
                    self.assertIn(name, argv)
                    instance = benchmark.load_attack(name, self.suite, None)
                    self.assertIsInstance(instance, attack_class)
                    self.assertEqual(getattr(instance, parameter), path)
                    self.assertIn("GOAL", instance.render("GOAL"))
                    raise SystemExit(0)

                with (
                    patch("run_agentdojo.os.open", return_value=99),
                    patch("run_agentdojo.os.dup2"),
                    patch("run_agentdojo.logging.basicConfig"),
                    patch.object(benchmark, "main", side_effect=benchmark_main) as entrypoint,
                    self.assertRaises(SystemExit) as raised,
                ):
                    run_case(args, ("banking", "user_task_1", "injection_task_0"))
                self.assertEqual(raised.exception.code, 0)
                entrypoint.assert_called_once()
                self.assertIs(benchmark.load_attack, original)


if __name__ == "__main__":
    unittest.main()
