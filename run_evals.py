# run_evals.py
# Automated Evaluation & Metrics Benchmark for Weekend Wizard
# Computes Tool Selection Accuracy, Schema Compliance Rate, Step Efficiency, and LLM-as-a-Judge Response Quality.

import asyncio
import json
import os
import sys
import time
from typing import Dict, Any, List
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Import core components from agent_fun
from agent_fun import client, MODEL, SYSTEM, llm_json, reflect

# ──────────────────────────────────────────────
# Evaluation Benchmark Test Cases
# ──────────────────────────────────────────────
TEST_CASES = [
    # ── Original test cases ──────────────────
    {
        "id": "TC_01_WEATHER",
        "category": "Parameter Inference",
        "query": "What is the weather like in Ahmedabad right now?",
        "expected_tools": ["get_weather"],
        "eval_guideline": "Verify the response mentions current weather conditions/temperature in Ahmedabad.",
    },
    {
        "id": "TC_02_BOOKS",
        "category": "Complex Schema Matching",
        "query": "Can you recommend some sci-fi books for me to read this weekend? I want to read a mystery thriller, give me 3 suggestions.",
        "expected_tools": ["book_recs"],
        "eval_guideline": "Verify the response recommends sci-fi/mystery thriller books and adheres to the requested quantity.",
    },
    {
        "id": "TC_03_MULTI_TOOL",
        "category": "Multi-Tool Chains",
        "query": "Tell me a quick joke to start the weekend! Can you show me a cute dog photo?",
        "expected_tools": ["random_joke", "random_dog"],
        "eval_guideline": "Verify the response includes both a joke and a link/reference to a dog photo.",
    },
    {
        "id": "TC_04_TRIVIA",
        "category": "Interactive Prompting",
        "query": "Give me a trivia question to test my knowledge!",
        "expected_tools": ["trivia"],
        "eval_guideline": "Verify the response asks a trivia question with multiple choice options and does NOT give away the final answer immediately.",
    },
    {
        "id": "TC_05_CONVERSATION",
        "category": "Casual Chat & Reflection",
        "query": "hello! I need help planning my weekend.",
        "expected_tools": [],
        "eval_guideline": "Verify the response is a welcoming, enthusiastic greeting asking how to help plan the weekend. No tools should have been called.",
    },
    # ── New edge-case test cases ─────────────
    {
        "id": "TC_06_AMBIGUOUS",
        "category": "Ambiguous Input",
        "query": "What's the weather?",
        "expected_tools": ["get_weather"],
        "eval_guideline": "Verify the agent either infers a default/popular location or asks for clarification, and provides weather data. The key test is that it handles the missing city gracefully rather than crashing.",
    },
    {
        "id": "TC_07_THREE_TOOL_CHAIN",
        "category": "3-Tool Chain",
        "query": "Tell me a joke, show me a dog photo, and give me a trivia question!",
        "expected_tools": ["random_joke", "random_dog", "trivia"],
        "eval_guideline": "Verify the response includes all three: a joke, a dog photo URL, and a trivia question with options. All three tools must have been called.",
    },
    {
        "id": "TC_08_OFF_TOPIC",
        "category": "Off-Topic Handling",
        "query": "What's 2+2?",
        "expected_tools": [],
        "eval_guideline": "Verify the agent answers the math question directly (answer: 4) without calling any tools. The agent should handle simple factual questions on its own.",
    },
    {
        "id": "TC_09_ERROR_RESILIENCE",
        "category": "Error Resilience",
        "query": "What's the weather in asdfghjkl?",
        "expected_tools": ["get_weather"],
        "eval_guideline": "Verify the agent attempts to get weather data (it may guess coordinates or handle the error). The key test is that it does not crash and provides a reasonable response.",
    },
    {
        "id": "TC_10_GREETING",
        "category": "Greeting (No Tools)",
        "query": "Hey! Good morning!",
        "expected_tools": [],
        "eval_guideline": "Verify the response is a friendly morning greeting. No tools should be called -- the agent should respond conversationally without fetching any data.",
    },
]

# ──────────────────────────────────────────────
# LLM-as-a-Judge Evaluator
# ──────────────────────────────────────────────
def evaluate_response_quality(query: str, answer: str, guideline: str) -> Dict[str, Any]:
    """Uses Groq LLM-as-a-Judge to evaluate the quality of the final response (1-5 scale)."""
    prompt = (
        f"You are an expert AI evaluation judge.\n\n"
        f"User Query: {query}\n"
        f"Agent Answer: {answer}\n"
        f"Evaluation Guideline: {guideline}\n\n"
        f"Rate the Agent's Answer on a scale of 1 to 5 based on how well it satisfies the User Query and adheres to the Evaluation Guideline.\n"
        f"Output your evaluation in valid JSON format exactly as follows:\n"
        f'{{"score": <int 1-5>, "reasoning": "<brief explanation>"}}'
    )
    
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": "You output ONLY valid JSON in the exact schema requested."}, {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    
    txt = resp.choices[0].message.content.strip()
    try:
        return json.loads(txt)
    except Exception:
        # Fallback parsing
        import re
        match = re.search(r"\{.*\}", txt, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"score": 3, "reasoning": "Judge output could not be parsed."}

# ──────────────────────────────────────────────
# Main Evaluation Runner
# ──────────────────────────────────────────────
async def run_evals():
    print("=" * 60)
    print("  Running Weekend Wizard Automated Evals & Metrics...")
    print("=" * 60)
    
    server_path = "server_fun.py"
    exit_stack = AsyncExitStack()
    stdio = await exit_stack.enter_async_context(
        stdio_client(StdioServerParameters(command=sys.executable, args=[server_path]))
    )
    r_in, w_out = stdio
    session = await exit_stack.enter_async_context(ClientSession(r_in, w_out))
    await session.initialize()

    tools = (await session.list_tools()).tools
    tool_index = {t.name: t for t in tools}
    tool_defs = []
    for t in tools:
        tool_defs.append(f"- {t.name}: {t.description}\n  Schema: {json.dumps(t.inputSchema)}")
    tool_defs_str = "\n".join(tool_defs)
    system_prompt = f"{SYSTEM}\n\nAvailable Tools and their JSON schemas:\n{tool_defs_str}"

    total_test_cases = len(TEST_CASES)
    correct_tool_selections = 0
    total_tools_called = 0
    schema_compliant_calls = 0
    total_quality_score = 0
    total_steps = 0
    
    results_log = []

    try:
        for tc in TEST_CASES:
            print(f"\n  Running Eval: [{tc['id']}] - {tc['category']}")
            print(f"  Query: '{tc['query']}'")
            
            history = [{"role": "system", "content": system_prompt}, {"role": "user", "content": tc["query"]}]
            
            tools_called = []
            schema_errors = 0
            start_time = time.time()
            final_answer = ""
            steps_taken = 0
            
            for step in range(8):
                steps_taken += 1
                try:
                    decision = llm_json(history)
                except Exception as e:
                    final_answer = f"Error parsing JSON: {e}"
                    break
                
                if decision.get("action") == "final":
                    final_answer = decision.get("answer", "")
                    final_answer = reflect(final_answer)
                    break
                
                tname = decision.get("action", "")
                args = decision.get("args", {})
                
                if tname in tool_index:
                    tools_called.append(tname)
                    total_tools_called += 1
                    print(f"    Called: {tname}({args})")
                    try:
                        result = await session.call_tool(tname, args)
                        payload = result.content[0].text if result.content else result.model_dump_json()
                        schema_compliant_calls += 1
                    except Exception as e:
                        payload = json.dumps({"error": str(e)})
                        schema_errors += 1
                else:
                    payload = f"System error: unknown tool '{tname}'."
                    schema_errors += 1
                
                observation = f"Tool result for {tname}: {payload}\nNow output the final answer using 'action': 'final'."
                history.append({"role": "user", "content": observation})
            
            elapsed = time.time() - start_time
            total_steps += steps_taken
            
            # Evaluate Tool Selection Accuracy
            missing_tools = [t for t in tc["expected_tools"] if t not in tools_called]
            extra_tools = [t for t in tools_called if t not in tc["expected_tools"]]

            # For test cases expecting NO tools, penalize if any were called
            if not tc["expected_tools"]:
                tsa_pass = len(tools_called) == 0
            else:
                tsa_pass = len(missing_tools) == 0

            if tsa_pass:
                correct_tool_selections += 1
            
            # Evaluate Response Quality via LLM Judge
            judge_res = evaluate_response_quality(tc["query"], final_answer, tc["eval_guideline"])
            score = judge_res.get("score", 0)
            total_quality_score += score
            
            results_log.append({
                "id": tc["id"],
                "category": tc["category"],
                "tools_called": tools_called,
                "expected_tools": tc["expected_tools"],
                "extra_tools_called": extra_tools,
                "tsa_pass": tsa_pass,
                "schema_errors": schema_errors,
                "judge_score": score,
                "judge_reasoning": judge_res.get("reasoning", ""),
                "latency_sec": round(elapsed, 2),
                "steps": steps_taken
            })
            
            print(f"  -- Final Answer: {final_answer[:100]}...")
            print(f"  -- Judge Score: {score}/5 | Reasoning: {judge_res.get('reasoning')}")
            tsa_label = "PASS" if tsa_pass else "FAIL"
            extra_note = f" (extra: {extra_tools})" if extra_tools else ""
            print(f"  -- TSA: {tsa_label}{extra_note} | Steps: {steps_taken} | Latency: {round(elapsed, 2)}s")

    finally:
        await exit_stack.aclose()
        
    # Calculate Aggregate Metrics
    tsa_metric = (correct_tool_selections / total_test_cases) * 100
    scr_metric = (schema_compliant_calls / total_tools_called * 100) if total_tools_called > 0 else 100.0
    avg_quality = total_quality_score / total_test_cases
    avg_steps = total_steps / total_test_cases
    
    print("\n" + "=" * 60)
    print("  EVALUATION METRICS SUMMARY")
    print("=" * 60)
    print(f"  Test Cases Run:                  {total_test_cases}")
    print(f"  1. Tool Selection Accuracy (TSA):  {tsa_metric:.1f}% ({correct_tool_selections}/{total_test_cases} passed)")
    print(f"  2. Schema Compliance Rate (SCR):   {scr_metric:.1f}% ({schema_compliant_calls}/{total_tools_called} valid calls)")
    print(f"  3. Response Quality (LLM Judge):   {avg_quality:.1f} / 5.0")
    print(f"  4. Step Efficiency (Avg Steps):    {avg_steps:.1f} steps per query")
    print("=" * 60)
    
    with open("eval_results_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "total_test_cases": total_test_cases,
                "tool_selection_accuracy": tsa_metric,
                "schema_compliance_rate": scr_metric,
                "avg_response_quality": avg_quality,
                "avg_steps": avg_steps
            },
            "detailed_results": results_log
        }, f, indent=2)
    print("  Detailed evaluation report saved to 'eval_results_report.json'.")

if __name__ == "__main__":
    asyncio.run(run_evals())
