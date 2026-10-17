import copy
import json
import unittest

from make_ill import INJECTION_POINT, build_case, close_result, compact


class ResultClosingTests(unittest.TestCase):
    def test_closes_body_and_preserves_trailing_fields(self):
        result = json.dumps(
            {
                "id": "message-id",
                "body": "visible email text " + INJECTION_POINT + " remaining email text",
                "attachments": [],
                "metadata": {"status": "keep"},
            },
            ensure_ascii=False,
            indent=2,
        )

        closed = close_result(result)

        self.assertTrue(closed.startswith(result.split(INJECTION_POINT)[0]))
        self.assertEqual(
            json.loads(closed),
            {
                "id": "message-id",
                "body": "visible email text ",
                "attachments": [],
                "metadata": {"status": "keep"},
            },
        )

    def test_escaped_quotes_backslashes_controls_and_unicode(self):
        prefix = '中文 "quoted" \\folder\\file\n\t\r\b\f\x01 '
        suffix = ' trailing "escaped quote" \\ path\nUnicode 日本語'
        for ensure_ascii in (False, True):
            with self.subTest(ensure_ascii=ensure_ascii):
                result = json.dumps(
                    {"body": prefix + INJECTION_POINT + suffix, "tail": 'keep "quote" \\'},
                    ensure_ascii=ensure_ascii,
                    indent=2,
                )

                closed = close_result(result)

                self.assertTrue(closed.startswith(result.split(INJECTION_POINT)[0]))
                self.assertEqual(json.loads(closed), {"body": prefix, "tail": 'keep "quote" \\'})

    def test_root_array_nested_objects_and_following_values(self):
        original = [
            {"before": 1},
            {"nested": [{"body": "head" + INJECTION_POINT + "tail", "after": [2, 3]}]},
            {"later": "preserve"},
        ]
        result = compact(original)

        closed = close_result(result)

        expected = copy.deepcopy(original)
        expected[1]["nested"][0]["body"] = "head"
        self.assertTrue(closed.startswith(result.split(INJECTION_POINT)[0]))
        self.assertEqual(json.loads(closed), expected)

    def test_root_json_string(self):
        result = compact('head "quoted" ' + INJECTION_POINT + 'tail \\')

        self.assertEqual(json.loads(close_result(result)), 'head "quoted" ')

    def test_closes_json_encoded_in_another_json_string(self):
        inner = json.dumps(
            {
                "body": 'head "quoted" \\path\n中文 ' + INJECTION_POINT + ' tail "quoted"',
                "attachments": [],
            },
            ensure_ascii=False,
            indent=2,
        )
        result = compact({"data": inner, "outer_tail": {"status": "preserve"}})

        closed = close_result(result)

        self.assertTrue(closed.startswith(result.split(INJECTION_POINT)[0]))
        outer = json.loads(closed)
        self.assertEqual(outer["outer_tail"], {"status": "preserve"})
        self.assertEqual(
            json.loads(outer["data"]),
            {"body": 'head "quoted" \\path\n中文 ', "attachments": []},
        )

    def test_closes_multiple_json_string_encoding_layers(self):
        inner = compact({"body": "head" + INJECTION_POINT + "tail", "attachments": []})
        result = compact({"data": compact({"nested": inner, "middle_tail": True}), "outer_tail": []})

        closed = close_result(result)

        self.assertTrue(closed.startswith(result.split(INJECTION_POINT)[0]))
        outer = json.loads(closed)
        middle = json.loads(outer["data"])
        self.assertEqual(outer["outer_tail"], [])
        self.assertTrue(middle["middle_tail"])
        self.assertEqual(json.loads(middle["nested"]), {"body": "head", "attachments": []})

    def test_complete_json_prefix_with_marker_at_end(self):
        for prefix in (compact({"body": "head", "attachments": []}), compact([{"body": "head"}])):
            with self.subTest(prefix=prefix):
                self.assertEqual(close_result(prefix + INJECTION_POINT), prefix)

    def test_plain_text_and_marker_boundaries(self):
        for result, expected in (
            ("plain head" + INJECTION_POINT + "plain tail", "plain head"),
            ('text "quoted" \\ ' + INJECTION_POINT + "tail", 'text "quoted" \\ '),
            (INJECTION_POINT + "tail", ""),
            ("head" + INJECTION_POINT, "head"),
            (INJECTION_POINT, ""),
        ):
            with self.subTest(result=result):
                self.assertEqual(close_result(result), expected)


class CaseClosingTests(unittest.TestCase):
    def test_nested_result_closes_before_model_and_task_and_reopens_original(self):
        original_result = json.dumps(
            {
                "id": "message-id",
                "body": "EMAIL_PREFIX " + INJECTION_POINT + " unchanged email tail",
                "attachments": [{"filename": "keep.txt"}],
            },
            ensure_ascii=False,
            indent=2,
        )
        response = {
            "parts": [{"functionResponse": {"name": "read_email", "response": {"result": original_result}}}],
            "role": "user",
        }
        model = {"parts": [{"text": "BENIGN_MODEL_SUMMARY"}], "role": "model"}
        case = {
            "contents": [
                {"parts": [{"text": "read email"}], "role": "user"},
                response,
                model,
            ],
            "task": "USER_TASK_ANCHOR",
        }
        original = copy.deepcopy(case)

        target, injection = build_case(case)

        result_path = ["contents", 1, "parts", 0, "functionResponse", "response", "result"]
        self.assertEqual(injection.path, result_path)
        self.assertEqual(case, original)
        closed = target["contents"][1]["parts"][0]["functionResponse"]["response"]["result"]
        self.assertEqual(
            json.loads(closed),
            {"id": "message-id", "body": "EMAIL_PREFIX ", "attachments": [{"filename": "keep.txt"}]},
        )
        self.assertEqual(target["contents"][2], model)
        self.assertEqual(target["contents"][3], {"parts": [{"text": "USER_TASK_ANCHOR"}], "role": "user"})
        resumed = target["contents"][4]["parts"][0]["functionResponse"]["response"]["result"]
        self.assertEqual(resumed, original_result.replace(INJECTION_POINT, ""))

        before, after = original_result.split(INJECTION_POINT)
        actual_result = injection.body["contents"][1]["parts"][0]["functionResponse"]["response"]["result"]
        decoded = json.loads(actual_result)
        self.assertEqual(decoded, {
            "id": "message-id",
            "body": "EMAIL_PREFIX " + injection.payload + " unchanged email tail",
            "attachments": [{"filename": "keep.txt"}],
        })
        self.assertEqual(actual_result, before + compact(injection.payload)[1:-1] + after)
        self.assertIn("BENIGN_MODEL_SUMMARY", injection.payload)
        self.assertIn("USER_TASK_ANCHOR", injection.payload)
        expected_ill = {"contents": copy.deepcopy(original["contents"][:2])}
        expected_ill["contents"][1]["parts"][0]["functionResponse"]["response"]["result"] = actual_result
        self.assertEqual(injection.body, expected_ill)


if __name__ == "__main__":
    unittest.main()
