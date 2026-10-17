import copy
import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from injector import loads
from make_ill import _closest_value, build_injection, compact, edit_distance, normalize


def reference_distance(left, right):
    """Independent full-matrix reference for the optimizer's edit-distance objective."""
    matrix = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(len(left) + 1):
        matrix[i][0] = i
    for j in range(len(right) + 1):
        matrix[0][j] = j
    for i, a in enumerate(left, 1):
        for j, b in enumerate(right, 1):
            matrix[i][j] = min(
                matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + (a != b)
            )
    return matrix[-1][-1]


def request(value, part_index=0):
    parts = [{"text": "another part"}] * part_index
    parts.append({"functionResponse": {"name": "read_email", "response": {"result": value}}})
    return {"contents": [{"parts": parts, "role": "user"}], "generationConfig": {"temperature": 0}}


class InjectionTests(unittest.TestCase):
    def test_marker_replacement_preserves_surrounding_text(self):
        before, after = 'fixed "prefix"\\\n ', ' \t\\"suffix"'
        for payload in ["  hello world  ", '中文 "quote" \\ slash\n', "\x00\x1f", ""]:
            with self.subTest(payload=payload):
                template = request(before + "[INJECTION POINT]" + after)
                target = request(before + payload + after)
                original = copy.deepcopy(template)
                result = build_injection(template, target)
                self.assertEqual(result.body, target)
                self.assertEqual(result.payload, payload)
                self.assertEqual(result.distance, 0)
                self.assertEqual(template, original)

    def test_marker_takes_priority_over_last_response_and_text(self):
        template = request("prefix [INJECTION POINT] suffix")
        template["contents"] += request("last response")["contents"]
        template["messages"] = [{"content": [{"text": "last text"}]}]
        target = copy.deepcopy(template)
        target["contents"][0]["parts"][0]["functionResponse"]["response"]["result"] = (
            "prefix inserted suffix"
        )
        result = build_injection(template, target)
        self.assertEqual(result.path[1], 0)
        self.assertEqual(result.payload, "inserted")
        self.assertEqual(result.body, target)

    def test_marker_in_arbitrary_nested_string(self):
        template = {"nested": [{"value": "a[INJECTION POINT]z"}], "fixed": True}
        target = {"nested": [{"value": "abcz"}], "fixed": True}
        result = build_injection(template, target)
        self.assertEqual(result.path, ["nested", 0, "value"])
        self.assertEqual(result.payload, "bc")
        self.assertEqual(result.body, target)

    def test_multiple_markers_require_an_unambiguous_path(self):
        template = {"x": "a[INJECTION POINT]z", "y": "[INJECTION POINT]"}
        target = {"x": "abz", "y": "[INJECTION POINT]"}
        with self.assertRaisesRegex(ValueError, "multiple fields"):
            build_injection(template, target)
        result = build_injection(template, target, ["x"])
        self.assertEqual(result.body, target)
        self.assertEqual(result.payload, "b")
        with self.assertRaisesRegex(ValueError, "at most one"):
            build_injection({"x": "[INJECTION POINT][INJECTION POINT]"}, target)

    def test_marker_optimizer_with_different_fixed_text(self):
        template = {"x": 'a"[INJECTION POINT]\\z', "fixed": 0}
        target = {"x": 'b"ab\\y', "fixed": 1}
        result = build_injection(template, target)
        self.assertEqual(result.body["x"], 'a"' + result.payload + '\\z')
        self.assertEqual(result.body["fixed"], 0)
        measured = reference_distance(normalize(compact(result.body)), normalize(compact(target)))
        self.assertEqual(result.distance, measured)
        candidates = [
            "".join(chars)
            for length in range(4)
            for chars in itertools.product('ab"\\', repeat=length)
        ]
        brute = min(
            reference_distance(
                normalize(compact({"x": 'a"' + payload + '\\z', "fixed": 0})),
                normalize(compact(target)),
            )
            for payload in candidates
        )
        self.assertLessEqual(result.distance, brute)

    def test_exact_match_for_string_only_changes(self):
        for value in ["  hello world  ", '中文 "quote" \\ slash\nline\t', "\x00\x08\x1f", ""]:
            with self.subTest(value=value):
                template, target = (
                    {"nested": [{"value": "old"}], "fixed": True},
                    {"nested": [{"value": value}], "fixed": True},
                )
                template_before, target_before = copy.deepcopy(template), copy.deepcopy(target)
                result = build_injection(template, target, ["nested", 0, "value"])
                self.assertEqual(result.distance, 0)
                self.assertEqual(result.body, target)
                self.assertEqual(result.payload, value)
                self.assertEqual(template, template_before)
                self.assertEqual(target, target_before)

    def test_added_turns_stay_inside_one_string(self):
        template = request('benign text with "}}} inside')
        target = copy.deepcopy(template)
        target["contents"] += [
            {"parts": [{"text": "a model response"}], "role": "model"},
            {"parts": [{"text": "1+1=?"}], "role": "user"},
        ]
        result = build_injection(template, target)
        actual = json.loads(compact(result.body))
        self.assertEqual(len(actual["contents"]), 1)
        self.assertGreater(result.distance, 0)
        self.assertLess(result.distance, result.baseline_distance)
        measured = reference_distance(normalize(compact(actual)), normalize(compact(target)))
        self.assertEqual(result.distance, measured)
        restored = copy.deepcopy(result.body)
        restored["contents"][0]["parts"][0]["functionResponse"]["response"]["result"] = (
            'benign text with "}}} inside'
        )
        self.assertEqual(restored, template)

    def test_default_path_searches_every_part(self):
        template = request("old", part_index=1)
        target = request("new", part_index=1)
        result = build_injection(template, target)
        self.assertEqual(
            result.path, ["contents", 0, "parts", 1, "functionResponse", "response", "result"]
        )
        self.assertEqual(result.body, target)

    def test_default_path_uses_last_response(self):
        template = request("first")
        template["contents"] += request("last")["contents"]
        target = copy.deepcopy(template)
        target["contents"][1]["parts"][0]["functionResponse"]["response"]["result"] = "changed"
        result = build_injection(template, target)
        self.assertEqual(result.path[1], 1)
        self.assertEqual(result.body, target)

    def test_target_differences_outside_the_field_are_scored(self):
        template, target = {"x": "old", "fixed": 0}, {"fixed": 1, "x": "new"}
        result = build_injection(template, target, ["x"])
        self.assertEqual(list(result.body), ["x", "fixed"])
        self.assertEqual(result.body["fixed"], 0)
        self.assertGreater(result.distance, 0)
        self.assertEqual(
            result.distance,
            reference_distance(normalize(compact(result.body)), normalize(compact(target))),
        )

    def test_duplicate_keys_survive(self):
        template = loads('{"x":"old","fixed":1,"fixed":2}')
        target = loads('{"x":"new","fixed":1,"fixed":2}')
        result = build_injection(template, target, ["x"])
        self.assertEqual(compact(result.body), compact(target))
        self.assertEqual(result.distance, 0)

    def test_invalid_paths_and_nonstring_fields(self):
        for path in [[], "x", [True], ["missing"], ["x", 99]]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                build_injection({"x": "old"}, {"x": "new"}, path)
        with self.assertRaises(TypeError):
            build_injection({"x": 1}, {"x": 2}, ["x"])
        with self.assertRaises(ValueError):
            build_injection({"x": "old"}, {"x": "new"})

    def test_cli_writes_the_entire_replacement_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            template, target = directory / "template.json", directory / "target.json"
            output, payload = directory / "ill.json", directory / "payload.txt"
            template.write_text('{"x":"old"}', encoding="utf-8")
            target.write_text('{"x":"new value"}', encoding="utf-8")
            process = subprocess.run(
                [
                    sys.executable,
                    "make_ill.py",
                    "--template",
                    str(template),
                    "--target",
                    str(target),
                    "--path",
                    '["x"]',
                    "--output",
                    str(output),
                    "--payload",
                    str(payload),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(json.loads(output.read_text()), {"x": "new value"})
            self.assertEqual(payload.read_text(), "new value")
            self.assertIn("-> 0 (minimum)", process.stdout)

    def test_cli_writes_only_marker_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            template, target = directory / "template.json", directory / "target.json"
            output, payload = directory / "ill.json", directory / "payload.txt"
            template.write_text('{"x":"prefix [INJECTION POINT] suffix"}', encoding="utf-8")
            target.write_text('{"x":"prefix new value suffix"}', encoding="utf-8")
            subprocess.run(
                [
                    sys.executable, "make_ill.py", "--template", str(template),
                    "--target", str(target), "--output", str(output), "--payload", str(payload),
                ],
                capture_output=True, text=True, check=True,
            )
            self.assertEqual(json.loads(output.read_text()), {"x": "prefix new value suffix"})
            self.assertEqual(payload.read_text(), "new value")


class OptimizerTests(unittest.TestCase):
    def test_distance_matches_independent_reference(self):
        cases = [
            ("", ""),
            ("abc", "adc"),
            ("abab", "baba"),
            ("xaaaaay", "xaaay"),
            ("你好", "你们好"),
        ]
        for left, right in cases:
            with self.subTest(left=left, right=right):
                self.assertEqual(edit_distance(left, right), reference_distance(left, right))

    def test_optimizer_against_exhaustive_small_candidates(self):
        # Enumerate decoded strings independently of the JSON-string automaton. Short
        # targets cover quote escaping, literal backslashes, and fixed-suffix alignment.
        alphabet = ["a", '"', "\\", "\n", "}"]
        candidates = [
            "".join(chars)
            for length in range(4)
            for chars in itertools.product(alphabet, repeat=length)
        ]
        targets = [
            "".join(chars)
            for length in range(3)
            for chars in itertools.product('a"\\}', repeat=length)
        ]
        for prefix, suffix in [("", ""), ("a", "}"), ('{"x":"', '"}')]:
            for target in targets:
                with self.subTest(prefix=prefix, suffix=suffix, target=target):
                    payload, minimum = _closest_value(prefix, target, suffix, [0] * len(target))
                    actual = reference_distance(prefix + compact(payload)[1:-1] + suffix, target)
                    brute = min(
                        reference_distance(prefix + compact(candidate)[1:-1] + suffix, target)
                        for candidate in candidates
                    )
                    self.assertEqual(actual, minimum)
                    self.assertLessEqual(minimum, brute)


if __name__ == "__main__":
    unittest.main()
