"""Send a saved generateContent request body to Gemini and print the reply.

    python ask_request.py tmp.json
    python ask_request.py tmp.json --model gemini-3.5-flash
    python ask_request.py tmp.json --json        # the whole response as JSON

The file is a request body in the API's wire format (contents, systemInstruction, tools,
generationConfig), e.g. a miniagent session's final request copied from the run viewer.
It is sent as is through gemini.ask_request(); edit it first to try variations.
"""

import argparse
import json

import gemini


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request", help="request body JSON file")
    ap.add_argument("--model", default=gemini.MODEL, help=f"model (default: {gemini.MODEL})")
    ap.add_argument("--json", action="store_true", help="print the whole response as JSON")
    args = ap.parse_args()

    resp = gemini.ask_request(args.request, model=args.model)
    if args.json:
        print(resp.model_dump_json(indent=2, exclude_none=True))
        return

    cand = resp.candidates[0] if resp.candidates else None
    if cand is None or cand.content is None:
        reason = getattr(cand, "finish_reason", None) or resp.prompt_feedback
        print(f"<no response: {reason}>")
        return
    for part in cand.content.parts or []:
        if part.function_call:   # with tools declared, the model may call one instead of answering
            fc = part.function_call
            print(f"→ {fc.name}({json.dumps(fc.args or {}, ensure_ascii=False)})")
        elif part.text:
            print(("[thought] " if part.thought else "") + part.text)

    usage = resp.usage_metadata
    tokens = f" · prompt {usage.prompt_token_count} tokens" if usage and usage.prompt_token_count else ""
    finish = getattr(cand.finish_reason, "value", cand.finish_reason)
    print(f"\n[{args.model} · finish {finish}{tokens}]")


if __name__ == "__main__":
    main()
