# structural-attack

Structural injection experiments on LLM agent tool results.

## Layout

```
harness/        agent: Gemini + Gmail/Calendar tools
  miniagent.py    terminal REPL agent
  tools.py        Gmail / Calendar tools
  google_auth.py  Google OAuth setup
  runlog.py       session logs -> runs/miniagent/
ui/             local web viewers
  viewer.py       agent session logs (runs/miniagent/)
  caseviewer.py   test cases (cases/), with "Ask Gemini"
  webui.py        shared server / page code
injector.py     ToolCase + build injected cases
gemini.py       one-shot Gemini call for case testing
ask_request.py  case json -> json with gemini
make_ill.py     case to case injection
run_agentdojo.py agentdojo wrapper with parallel execution
cases/          *_template / *_target / *_ill.json
notebooks/
```

## Setup

```bash
poetry install
echo "GEMINI_API_KEY=..." > .env
```

Run everything from the repo root.

## Run the agent

```bash
python -m harness.google_auth   # once; needs ~/.gmail-mcp/gcp-oauth.keys.json
python -m harness.miniagent     # chat in the terminal, /quit to exit
python -m ui.viewer             # http://localhost:8765, live session logs
```

## Test with cases

A case is `{call, body, question}`: tool call, tool result records, user question.
`template` is the benign result, `target` is what the attacker wants the model to see,
and `ill` is the template with the payload injected into one string field.

```bash
python injector.py              # template + target -> cases/email_ill.json (edit __main__ to change)
python -m ui.caseviewer         # http://localhost:8766
```


# Experimental Setup
Tested on AgentDojo benchmark v1.2.2 (full); the runner uses this fixed version.

| Environment | Tools | User tasks | Injection tasks | Cross cases (User × Injection) |
|---|---|---|---|---|
| Workspace | 24 | 40 | 14 | 560 |
| Slack | 11 | 21 | 5 | 105 |
| Travel | 28 | 20 | 7 | 140 |
| Banking | 11 | 16 | 9 | 144 |
| **Total** | 74 | **97** | **35** | **949** |

Runs per setting: benign = 97; with attack = 949 cross cases + 35 injection tasks run alone (injection-task utility check) = 984.

test sample command for qwen3
```
benign
python run_agentdojo.py \
  --model OPENAI_COMPATIBLE \
  --model-id qwen/qwen3-235b-a22b-2507 \
  --logdir runs/qwen3-235b-a22b-2507
  --attack important_instructions / ours / others
```

`my_attack` reads benign logs from `runs/<model-name>/openai-compatible/<suite>/<user_task>/none/none.json`
and converts messages to API format in memory, solely to select the benign turns to replay.
Closing and reopening tool responses use independent local ground-truth simulations cached in
`mocked_tool_response/v1.2.2/<suite>/<user_task>.json`. All matching tool calls are
cached; the first response per injection vector supplies its boundaries. If the benign run never
reads a vector, its final assistant reply is replayed with the same mocked boundaries.

```bash
.venv/bin/python run_agentdojo.py \
  --model-id qwen/qwen3-235b-a22b-2507 \
  --logdir runs/qwen3-235b-a22b-2507_myattack_with_bait \
  -ml my_attacks --attack my_attack
```

Set `MY_ATTACK_BENIGN_DIR` to override the directory containing the suite folders
(for example, `runs/qwen3-235b-a22b-2507/openai-compatible`).

**metrics:**
utility: if the benign task completed
security: if the malicious task are completed



# Eval result


| Overall | Benign | Important Instr | ChatInject | MyAttack_withlog |
|---|---|---|---|---|
| ASR ↑ | – | 56.3 (534/949) | 35.2 (334/949) | |
| Utility ↓ | 79.4 (77/97) | 33.8 (321/949) | 43.4 (412/949) | |
