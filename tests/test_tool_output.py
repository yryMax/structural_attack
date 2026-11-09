import json
import unittest
from types import SimpleNamespace

import yaml
from pydantic import BaseModel

from agentdojo.agent_pipeline.ground_truth_pipeline import tool_result_to_str as ground_truth_formatter
from agentdojo.agent_pipeline.tool_execution import ToolsExecutor, tool_result_to_str
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

from my_attacks import OneTurnAttack
from run_agentdojo import disable_tool_output_wrapping


class Record(BaseModel):
    body: str


class ToolOutputTests(unittest.TestCase):
    def setUp(self):
        defaults = tool_result_to_str.__defaults__
        self.addCleanup(setattr, tool_result_to_str, "__defaults__", defaults)
        disable_tool_output_wrapping()

    def test_removes_inserted_folds_and_preserves_string_contents(self):
        record = Record(body='"quoted" \\path\n' + 'long text with spaces ' * 30 + '\n中文\tend')
        for result in (record, [record]):
            with self.subTest(result_type=type(result)):
                upstream = tool_result_to_str(result, dump_fn=yaml.safe_dump)
                corrected = tool_result_to_str(result)
                self.assertIn("\\\n", upstream)
                self.assertNotIn("\\\n", corrected)
                self.assertEqual(yaml.safe_load(corrected), yaml.safe_load(upstream))
        self.assertIs(ground_truth_formatter, tool_result_to_str)
        self.assertIs(ToolsExecutor().output_formatter, tool_result_to_str)
        self.assertEqual(json.loads(tool_result_to_str(record, dump_fn=json.dumps)), record.model_dump())

    def test_injected_transactions_have_no_backslash_continuations(self):
        suite = get_suite("v1.2.2", "banking")
        task = suite.user_tasks["user_task_1"]
        goal = 'A long goal with "quotes", \\path and 中文. ' * 5
        injections = OneTurnAttack(suite, None).attack(task, SimpleNamespace(GOAL=goal))
        payload = json.loads('"' + injections["injection_incoming_transaction"] + '"')
        environment = task.init_environment(suite.load_and_inject_default_environment(injections))
        call = FunctionCall(function="get_most_recent_transactions", args={"n": 5}, id="transactions")
        _, _, _, messages, _ = ToolsExecutor().query(
            task.PROMPT, FunctionsRuntime(suite.tools), environment,
            [{"role": "assistant", "content": None, "tool_calls": [call]}],
        )
        self.assertIsNone(messages[-1]["error"])
        text = messages[-1]["content"][0]["content"]
        self.assertIn("<turn_start><tool_response>- amount: 100.0", text)
        self.assertNotIn("\\\n", text)
        self.assertEqual(yaml.safe_load(text)[-1]["subject"], payload)
        self.assertIn("<turn_start><user>" + goal + "</><turn_end>", payload)

    def test_copied_workspace_boundaries_have_no_backslash_continuations(self):
        suite = get_suite("v1.2.2", "workspace")
        attack = OneTurnAttack(suite, None)
        boundaries = attack._get_tool_boundaries(suite.user_tasks["user_task_35"])
        self.assertEqual(len(boundaries), 4)
        for prefix, suffix in boundaries.values():
            self.assertNotIn("\\\n", prefix + suffix)


if __name__ == "__main__":
    unittest.main()
