import json
import os
from typing import Any, Dict, List
import boto3

# Bedrock Agent Runtime client for KB Retrieve
_region = boto3.session.Session().region_name or os.environ.get("AWS_REGION", "us-east-1")
AGENT_RT = boto3.client("bedrock-agent-runtime", region_name=_region)

KB_ID = os.environ["KNOWLEDGE_BASE_ID"]
TOP_K = int(os.environ.get("NUM_RESULTS", "5"))

def _parse_body(event: Dict[str, Any]) -> Dict[str, Any]:
    if "body" in event:
        try:
            if isinstance(event["body"], (dict, list)):
                return event["body"]
            return json.loads(event["body"] or "{}")
        except Exception:
            return {}
    return event if isinstance(event, dict) else {}

def lambda_handler(event, context):
    body = _parse_body(event)
    city = (body.get("city") or "").strip() if isinstance(body.get("city"), str) else None
    question = (body.get("question") or "").strip() if isinstance(body.get("question"), str) else ""

    if not question and city:
        question = f"events in {city}"
    if not question and not city:
        return {"statusCode": 400, "body": json.dumps({"error": "Provide 'city' and/or free-form 'question'."})}

    query_text = question if question else f"events in {city}"

    try:
        resp = AGENT_RT.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query_text},
            retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": TOP_K}},
        )
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": "Retrieve call failed", "details": str(e)})}

    results: List[Dict[str, Any]] = resp.get("retrievalResults", []) or []
    if not results:
        msg = f"No upcoming events found for {city or 'the query'}."
        return {"statusCode": 200, "body": json.dumps({"events": msg})}

    lines: List[str] = []
    for hit in results[:3]:
        content = hit.get("content") or {}
        txt = (content.get("text") or "").strip()
        if not txt:
            continue
        # concise preview
        lines.append(txt[:200] + ("…" if len(txt) > 200 else ""))

    if not lines:
        msg = f"No upcoming events found for {city or 'the query'}."
    else:
        msg = "\n".join(lines)

    return {"statusCode": 200, "body": json.dumps({"events": msg})}
