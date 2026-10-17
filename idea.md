## Reference 
Security Challenges in AI Agent Deployment: Insights from a Large Scale Public Competition
https://arxiv.org/html/2507.20526v1


How Vulnerable Are AI Agents to Indirect Prompt Injections? Insights from a Large-Scale Public Competition
https://arxiv.org/abs/2603.15714


MCP Security Bench (MSB): Benchmarking Attacks Against Model Context Protocol in LLM Agents
https://arxiv.org/abs/2510.15994


### IDEA
在SDK层面inject！！！ 伪造一轮ai的回复，用正确的回答closing 然后开始harmful instr
role layer{
    one of:
        text layer {}
        function call layer {}
        function response layer{
            body layer {
                "email sender: " ...
                "email receiver:  " ...
                data layer {
                    ORIGINAL + OUR INJECTION POINT which is
                    closing body layer + closing response layer 
                    + forge another user task that ask for retrieving another email + malicious task
                    + forge the starting of the data layer

                }
            }
        }
    ...
}...



role layer{
    one of:
        text layer {}
        function call layer {}
        function response layer{
            body layer {
                "email sender: " ...
                "email receiver:  " ...
                data layer {
                    ORIGINAL + OUR INJECTION POINT which is
                    closing body layer + closing response layer 
                    + forge another user task that ask for retrieving another email + malicious task
                    + forge the starting of the data layer

                }
            }
        }
    ...
}...


effectivenes

1. payload calculation matters!（？！）
2. 合理的closing会导致大的提升！！！ matches response > err > irrelavant response (ablation)
3. pure string > seperatedly loaded (ask_request2 are more vulnerable than ask_request1)
```python
def ask_request(path, model: str = MODEL) -> types.GenerateContentResponse:
    """Send the saved request body in `path` (see load_request) and return the response."""
    contents, config = load_request(path)
    return _get_client().models.generate_content(model=model, contents=contents, config=config)


def ask_request2(path, model: str = MODEL) -> types.GenerateContentResponse:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    contents = json.dumps(body.get("contents", []), ensure_ascii=False)  
   # print(contents)
    return _get_client().models.generate_content(
            model=model,
            contents=contents,
            config=None,
        )
```

