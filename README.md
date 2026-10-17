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

AgentDojo benchmark `v1.2.2` (default). The paper (v1) has 6 workspace injection tasks; v1.2 raised it to 14.

| Environment | Tools | User tasks | Injection tasks | Cross cases (User × Injection) |
|---|---|---|---|---|
| Workspace | 24 | 40 | 14 | 560 |
| Slack | 11 | 21 | 5 | 105 |
| Travel | 28 | 20 | 7 | 140 |
| Banking | 11 | 16 | 9 | 144 |
| **Total** | 74 | **97** | **35** | **949** |

Runs per setting: benign = 97; with attack = 949 cross cases + 35 injection tasks run alone (injection-task utility check) = 984.

tested for qwen3
```
benign
python run_agentdojo.py \
  --model OPENAI_COMPATIBLE \
  --model-id qwen/qwen3-235b-a22b-2507 \
  --logdir runs/qwen3-235b-a22b-2507
  --attack important_instructions / ours / others
```

utility: if the benign task completed
security: if the malicious task are completed



# Eval result