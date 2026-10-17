import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from injector import DupDict, loads
from make_ill import INJECTION_POINT, build_case, build_target, compact


SCRIPT = Path(__file__).resolve().parents[1] / "make_ill.py"
RESULT_PATH = ["contents", 1, "parts", 1, "functionResponse", "response", "result"]


def request(result="left[INJECTION POINT]right"):
    return {
        "contents": [
            {"parts": [{"text": "original request"}], "role": "user"},
            {
                "parts": [
                    {"text": "earlier part"},
                    {
                        "functionResponse": {
                            "name": "read_email",
                            "id": "call_123",
                            "response": {"result": result, "status": "ok"},
                        },
                        "partMetadata": "keep",
                    },
                    {"text": "later part"},
                ],
                "role": "user",
                "turnMetadata": {"source": "tool"},
            },
            {"parts": [{"text": "later turn"}], "role": "model"},
        ],
        "generationConfig": {"temperature": 0},
    }


def result_of(turn, part_index=0):
    return turn["parts"][part_index]["functionResponse"]["response"]["result"]


class TargetGenerationTests(unittest.TestCase):
    def test_reopened_response_repeats_prefix_and_preserves_parts_and_metadata(self):
        template = request('left "quoted"\n[INJECTION POINT]\\right 中文')
        template["contents"].append({"parts": [{"text": "independent later turn"}], "role": "user"})
        original = copy.deepcopy(template)
        task = '请处理 "这封邮件"'
        target = build_target(template, task)

        left = copy.deepcopy(original["contents"][1])
        left["parts"] = left["parts"][:2]
        left["parts"][1]["functionResponse"]["response"]["result"] = 'left "quoted"\n'
        resumed = copy.deepcopy(original["contents"][1])
        resumed["parts"][1]["functionResponse"]["response"]["result"] = 'left "quoted"\n\\right 中文'
        expected = copy.deepcopy(original)
        expected["contents"][1:] = [
            left,
            copy.deepcopy(original["contents"][2]),
            copy.deepcopy(original["contents"][3]),
            {"parts": [{"text": task}], "role": "user"},
            resumed,
        ]
        self.assertEqual(target, expected)
        self.assertEqual(template, original)

        # Closed/reopened responses and replayed turns are independent copies.
        target["contents"][1]["turnMetadata"]["source"] = "changed"
        target["contents"][2]["parts"][0]["text"] = "changed"
        target["contents"][3]["parts"][0]["text"] = "changed"
        self.assertEqual(template, original)
        self.assertEqual(target["contents"][5]["turnMetadata"]["source"], "tool")

    def test_empty_halves_are_preserved_at_marker_boundaries(self):
        for value, before, after in [
            (INJECTION_POINT + "tail", "", "tail"),
            ("head" + INJECTION_POINT, "head", ""),
            (INJECTION_POINT, "", ""),
        ]:
            with self.subTest(value=value):
                target = build_target(request(value), "task", RESULT_PATH)
                self.assertEqual(len(target["contents"]), 5)
                self.assertEqual(result_of(target["contents"][1], 1), before)
                self.assertEqual(result_of(target["contents"][4], 1), before + after)
                self.assertEqual(target["contents"][4]["parts"][0], request()["contents"][1]["parts"][0])
                self.assertEqual(target["contents"][4]["parts"][2], request()["contents"][1]["parts"][2])
                self.assertEqual(target["contents"][2], request()["contents"][2])
                self.assertEqual(target["contents"][3], {"parts": [{"text": "task"}], "role": "user"})

    def test_explicit_path_selects_one_of_multiple_response_markers(self):
        template = request()
        template["contents"].append(
            {"parts": [{"functionResponse": {"name": "other", "response": {"result": INJECTION_POINT}}}], "role": "user"}
        )
        with self.assertRaises(ValueError):
            build_target(template, "task")
        target = build_target(template, "task", RESULT_PATH)
        self.assertEqual(target["contents"][3], template["contents"][-1])
        self.assertEqual(result_of(target["contents"][1], 1), "left")
        self.assertEqual(result_of(target["contents"][-1], 1), "leftright")

    def test_invalid_marker_locations_and_counts(self):
        templates = [request("no marker"), request(INJECTION_POINT * 2)]
        wrong_field = request("no marker")
        wrong_field["contents"][0]["parts"][0]["text"] = INJECTION_POINT
        templates.append(wrong_field)
        for template in templates:
            with self.subTest(template=template), self.assertRaises((ValueError, TypeError)):
                build_target(template, "task")

    def test_invalid_paths_and_nonstring_results(self):
        for path in [
            [],
            "contents",
            ["contents", True, "parts", 1, "functionResponse", "response", "result"],
            ["contents", 99, "parts", 1, "functionResponse", "response", "result"],
            ["contents", 1, "parts", 99, "functionResponse", "response", "result"],
            ["contents", 1, "parts", 0, "text"],
            ["generationConfig", "temperature"],
            RESULT_PATH[:-1] + ["status"],
        ]:
            with self.subTest(path=path), self.assertRaises((ValueError, TypeError)):
                build_target(request(), "task", path)
        for value in [None, 1, [], {}]:
            with self.subTest(result=value), self.assertRaises((ValueError, TypeError)):
                build_target(request(value), "task", RESULT_PATH)

    def test_invalid_tasks_and_request_structure(self):
        for task in [None, 1, False, [], {}, ""]:
            with self.subTest(task=task), self.assertRaises((ValueError, TypeError)):
                build_target(request(), task)
        for template in [None, [], {}, {"contents": None}, {"contents": {"parts": []}}, {"contents": [None]}, {"contents": [{"parts": None}]}]:
            with self.subTest(template=template), self.assertRaises((ValueError, TypeError)):
                build_target(template, "task")


class CaseGenerationTests(unittest.TestCase):
    def test_case_injects_continuation_and_excludes_task_metadata(self):
        case = request('before "quoted"\n[INJECTION POINT] after\\tail')
        case["task"] = "执行这个任务"
        original = copy.deepcopy(case)
        target, injection = build_case(case)

        self.assertEqual(case, original)
        self.assertNotIn("task", target)
        self.assertNotIn("task", injection.body)
        self.assertEqual(injection.path, RESULT_PATH)
        self.assertEqual(result_of(injection.body["contents"][1], 1), 'before "quoted"\n' + injection.payload + " after\\tail")
        expected_ill = {key: copy.deepcopy(value) for key, value in original.items() if key != "task"}
        del expected_ill["contents"][2:]
        expected_ill["contents"][1]["parts"][1]["functionResponse"]["response"]["result"] = (
            original["contents"][1]["parts"][1]["functionResponse"]["response"]["result"].replace(INJECTION_POINT, injection.payload)
        )
        self.assertEqual(injection.body, expected_ill)
        self.assertEqual(len(injection.body["contents"]), len(original["contents"]) - 1)
        self.assertLessEqual(injection.distance, injection.baseline_distance)
        self.assertEqual(target["contents"][2], original["contents"][2])
        self.assertEqual(target["contents"][3], {"parts": [{"text": original["task"]}], "role": "user"})
        self.assertEqual(result_of(target["contents"][-1], 1), 'before "quoted"\n after\\tail')

    def test_entire_model_turn_moves_before_task_once_and_is_encoded_in_payload(self):
        case = request()
        model_turn = {
            "role": "model",
            "parts": [
                {"text": "MODEL_OUTPUT_FIRST", "thought": True, "thoughtSignature": "signature_first"},
                {"text": "MODEL_OUTPUT_SECOND", "thoughtSignature": "signature_second"},
                {"functionCall": {"name": "follow_up", "args": {"value": 3}}, "partMetadata": "keep model metadata"},
            ],
            "turnMetadata": {"source": "model", "sequence": 42},
        }
        case["contents"][2] = copy.deepcopy(model_turn)
        later_turns = [
            {"parts": [{"text": "independent user turn"}], "role": "user"},
            {"parts": [{"text": "independent model turn"}], "role": "model"},
        ]
        case["contents"].extend(copy.deepcopy(later_turns))
        case["task"] = "inserted task"
        original = copy.deepcopy(case)

        target, injection = build_case(case)

        self.assertEqual(case, original)
        self.assertEqual(target["contents"][2], model_turn)
        self.assertEqual(sum(turn == model_turn for turn in target["contents"]), 1)
        self.assertEqual(target["contents"][3:5], later_turns)
        self.assertEqual(target["contents"][5], {"parts": [{"text": case["task"]}], "role": "user"})
        self.assertEqual(result_of(target["contents"][6], 1), "leftright")
        self.assertNotIn(model_turn, injection.body["contents"])
        self.assertEqual(injection.body["contents"][2:], [])
        for text in ["MODEL_OUTPUT_FIRST", "MODEL_OUTPUT_SECOND", "signature_first", "signature_second", "independent user turn", "independent model turn"]:
            self.assertIn(text, injection.payload)
        self.assertLessEqual(injection.distance, injection.baseline_distance)

    def test_continuation_is_replayed_even_without_a_following_model_turn(self):
        for next_turn in [None, {"parts": [{"text": "following user"}], "role": "user"}]:
            with self.subTest(next_turn=next_turn):
                case = request()
                case["contents"] = case["contents"][:2]
                if next_turn is not None:
                    case["contents"].append(copy.deepcopy(next_turn))
                case["task"] = "inserted task"
                target, injection = build_case(case)
                self.assertEqual(target["contents"][2:-2], case["contents"][2:])
                self.assertEqual(target["contents"][-2], {"parts": [{"text": case["task"]}], "role": "user"})
                self.assertEqual(result_of(target["contents"][-1], 1), "leftright")
                self.assertEqual(injection.body["contents"][2:], [])
                self.assertEqual(len(injection.body["contents"]), 2)

    def test_negative_indices_select_original_response_before_truncation(self):
        case = request()
        case["task"] = "inserted task"
        path = ["contents", -2, "parts", -2, "functionResponse", "response", "result"]
        target, injection = build_case(case, path)

        self.assertEqual(target["contents"][2], case["contents"][2])
        self.assertEqual(target["contents"][3], {"parts": [{"text": case["task"]}], "role": "user"})
        self.assertEqual(injection.path, RESULT_PATH)
        self.assertEqual(result_of(target["contents"][-1], 1), "leftright")
        self.assertEqual(result_of(injection.body["contents"][1], 1), "left" + injection.payload + "right")
        selected_value = injection.body
        for key in injection.path:
            selected_value = selected_value[key]
        self.assertEqual(selected_value, "left" + injection.payload + "right")
        self.assertNotIn(case["contents"][2], injection.body["contents"])

    def test_screenshot_replays_execution_chain_and_reopens_complete_nested_json(self):
        nested_result = compact({
            "body": "BODY_PREFIX_ANCHOR " + INJECTION_POINT + " remaining email text",
            "attachments": [],
            "metadata": {"quoted": 'a "quote"', "path": "folder\\file"},
        })
        case = request(nested_result)
        continuation = [
            {"role": "model", "parts": [{"functionCall": {"name": "lookup", "id": "BENIGN_CALL_FIRST", "args": {}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "lookup", "response": {"result": "BENIGN_RETURN_FIRST"}}}]},
            {"role": "model", "parts": [{"functionCall": {"name": "summarize", "id": "BENIGN_CALL_SECOND", "args": {"item": 7}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "summarize", "response": {"result": "BENIGN_RETURN_SECOND"}}}]},
            {"role": "model", "parts": [{"text": "BENIGN_FINAL_ANSWER", "thoughtSignature": "CHAIN_SIGNATURE"}]},
        ]
        case["contents"][2:] = copy.deepcopy(continuation)
        case["systemInstruction"] = {"parts": [{"text": "keep system instruction"}]}
        case["tools"] = [{"functionDeclarations": [{"name": "lookup"}]}]
        case["task"] = "INSERTED_TASK_ANCHOR"
        original = copy.deepcopy(case)
        # The selected response is six turns from the end, with one later part.
        path = ["contents", -6, "parts", -2, "functionResponse", "response", "result"]

        target, injection = build_case(case, path)

        self.assertEqual(case, original)
        self.assertEqual(injection.path, RESULT_PATH)
        before, after = nested_result.split(INJECTION_POINT)
        self.assertEqual(
            json.loads(result_of(target["contents"][1], 1)),
            {
                "body": "BODY_PREFIX_ANCHOR ",
                "attachments": [],
                "metadata": {"quoted": 'a "quote"', "path": "folder\\file"},
            },
        )
        self.assertEqual(target["contents"][2:-2], continuation)
        for turn in continuation:
            self.assertEqual(sum(candidate == turn for candidate in target["contents"]), 1)
        self.assertEqual(target["contents"][-2], {"parts": [{"text": case["task"]}], "role": "user"})
        resumed = copy.deepcopy(original["contents"][1])
        resumed["parts"][1]["functionResponse"]["response"]["result"] = before + after
        self.assertEqual(target["contents"][-1], resumed)
        self.assertEqual(json.loads(result_of(target["contents"][-1], 1)), json.loads(nested_result.replace(INJECTION_POINT, "")))
        expected_ill = {key: copy.deepcopy(value) for key, value in original.items() if key != "task"}
        expected_ill["contents"] = expected_ill["contents"][:2]
        expected_result = json.loads(nested_result)
        expected_result["body"] = "BODY_PREFIX_ANCHOR " + injection.payload + " remaining email text"
        actual_result = result_of(injection.body["contents"][1], 1)
        self.assertEqual(json.loads(actual_result), expected_result)
        self.assertEqual(actual_result, before + compact(injection.payload)[1:-1] + after)
        expected_ill["contents"][1]["parts"][1]["functionResponse"]["response"]["result"] = actual_result
        self.assertEqual(injection.body, expected_ill)
        self.assertEqual(len(injection.body["contents"]), 2)
        for identifier in ["BENIGN_CALL_FIRST", "BENIGN_RETURN_FIRST", "BENIGN_CALL_SECOND", "BENIGN_RETURN_SECOND", "BENIGN_FINAL_ANSWER", "CHAIN_SIGNATURE", "INSERTED_TASK_ANCHOR", "BODY_PREFIX_ANCHOR"]:
            self.assertIn(identifier, injection.payload)
        for setting in ["generationConfig", "systemInstruction", "tools"]:
            self.assertEqual(target[setting], original[setting])
            self.assertEqual(injection.body[setting], original[setting])

    def test_marker_at_end_still_repeats_complete_nonempty_json_prefix(self):
        original_result = compact({"body": "NONEMPTY_COMPLETE_PREFIX", "attachments": []})
        case = request(original_result + INJECTION_POINT)
        case["contents"] = case["contents"][:2]
        case["task"] = "task"
        original = copy.deepcopy(case)

        target, injection = build_case(case)

        self.assertEqual(case, original)
        self.assertEqual(result_of(target["contents"][1], 1), original_result)
        self.assertEqual(result_of(target["contents"][-1], 1), original_result)
        self.assertEqual(json.loads(result_of(target["contents"][-1], 1)), {"body": "NONEMPTY_COMPLETE_PREFIX", "attachments": []})
        self.assertEqual(result_of(injection.body["contents"][1], 1), original_result + injection.payload)
        self.assertIn("NONEMPTY_COMPLETE_PREFIX", injection.payload)
        self.assertEqual(len(injection.body["contents"]), 2)

    def test_duplicate_keys_survive_and_all_top_level_tasks_are_removed(self):
        case = loads(
            '{"contents":[{"parts":[{"functionResponse":{"name":"old","name":"tool",'
            '"response":{"result":"a[INJECTION POINT]b","fixed":1,"fixed":2}}}],'
            '"role":"user"}],"task":"earlier task","fixed":1,"task":"任务","fixed":2}'
        )
        source_text = compact(case)
        target, injection = build_case(case)
        self.assertEqual(compact(case), source_text)
        self.assertIsInstance(target, DupDict)
        self.assertIsInstance(injection.body, DupDict)
        for output in [target, injection.body]:
            pairs = output.items()
            self.assertEqual([(key, value) for key, value in pairs if key == "fixed"], [("fixed", 1), ("fixed", 2)])
            self.assertNotIn("task", [key for key, _ in pairs])
            self.assertNotIn('"task":', compact(output))
            response = output["contents"][0]["parts"][0]["functionResponse"]
            self.assertEqual([(key, value) for key, value in response.items() if key == "name"], [("name", "old"), ("name", "tool")])
            self.assertEqual([(key, value) for key, value in response["response"].items() if key == "fixed"], [("fixed", 1), ("fixed", 2)])
        self.assertEqual(target["contents"][1]["parts"][0]["text"], "任务")

    def test_invalid_cases_require_a_task(self):
        for case in [None, [], {}, request()]:
            with self.subTest(case=case), self.assertRaises((ValueError, TypeError)):
                build_case(case)
        for task in [None, 1, False, [], {}, ""]:
            case = request()
            case["task"] = task
            with self.subTest(task=task), self.assertRaises((ValueError, TypeError)):
                build_case(case)

    def test_cli_explicit_case_roundtrip_with_unicode_and_quotes(self):
        case = request('开头 "quoted"[INJECTION POINT]结尾\\')
        case["task"] = '请转发 "这封邮件"\n第二行'
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            case_path = directory / "input.json"
            target_path = directory / "generated-target.json"
            ill_path = directory / "generated-ill.json"
            payload_path = directory / "generated-payload.txt"
            case_path.write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
            process = subprocess.run(
                [sys.executable, str(SCRIPT), "--case", str(case_path), "--target", str(target_path), "--output", str(ill_path), "--payload", str(payload_path)],
                cwd=directory,
                capture_output=True,
                text=True,
                check=True,
            )
            target = loads(target_path.read_text(encoding="utf-8"))
            ill = loads(ill_path.read_text(encoding="utf-8"))
            payload = payload_path.read_text(encoding="utf-8")
            self.assertEqual(target, build_target({key: value for key, value in case.items() if key != "task"}, case["task"]))
            self.assertEqual(result_of(ill["contents"][1], 1), '开头 "quoted"' + payload + "结尾\\")
            self.assertEqual(target["contents"][2], case["contents"][2])
            self.assertEqual(target["contents"][3], {"parts": [{"text": case["task"]}], "role": "user"})
            self.assertNotIn(case["contents"][2], ill["contents"])
            self.assertIn(case["contents"][2]["parts"][0]["text"], payload)
            self.assertNotIn("task", target)
            self.assertNotIn("task", ill)
            self.assertIn(str(target_path), process.stdout)

    def test_cli_defaults_read_case_json_in_working_directory(self):
        case = {
            "contents": [{"parts": [{"functionResponse": {"name": "tool", "response": {"result": "a[INJECTION POINT]b"}}}], "role": "user"}],
            "task": "任务",
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "case.json").write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
            subprocess.run([sys.executable, str(SCRIPT)], cwd=directory, capture_output=True, text=True, check=True)
            target = loads((directory / "target.json").read_text(encoding="utf-8"))
            ill = loads((directory / "ill.json").read_text(encoding="utf-8"))
            payload = (directory / "payload.txt").read_text(encoding="utf-8")
            self.assertEqual(target["contents"][1], {"parts": [{"text": "任务"}], "role": "user"})
            self.assertEqual(result_of(target["contents"][0]), "a")
            self.assertEqual(result_of(target["contents"][2]), "ab")
            self.assertEqual(result_of(ill["contents"][0]), "a" + payload + "b")
            self.assertNotIn("task", ill)


if __name__ == "__main__":
    unittest.main()
