import os
import json
import boto3
from datetime import datetime, timezone

agent_runtime = boto3.client('bedrock-agent-runtime')

def to_epoch_midnight(date_iso: str) -> float:
    dt = datetime.strptime(date_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc, hour=0, minute=0, second=0, microsecond=0)
    return float(dt.timestamp())

def parse_body(event):
    # Lambda Function URL may deliver body differently
    if isinstance(event, dict):
        if "body" in event and isinstance(event["body"], str):
            try:
                return json.loads(event["body"])
            except:
                return {}
        return event
    if isinstance(event, str):
        try:
            return json.loads(event)
        except:
            return {}
    return {}

def handler(event, context):
    kb_id = os.environ["KNOWLEDGE_BASE_ID"]
    default_top_k = int(os.environ.get("DEFAULT_TOP_K", "5"))

    body = parse_body(event)
    question  = body.get("question") or ""
    city      = body.get("city")
    from_date = body.get("from_date")
    top_k     = int(body.get("top_k") or default_top_k)

    # Build attribute filter
    filter_clause = None
    if city and from_date:
        filter_clause = {
            "andAll": [
                {"equals": {"key": "city", "value": city}},
                {"greaterThanOrEquals": {"key": "event_epoch", "value": to_epoch_midnight(from_date)}}
            ]
        }
    elif city:
        filter_clause = {"equals": {"key": "city", "value": city}}
    elif from_date:
        filter_clause = {"greaterThanOrEquals": {"key": "event_epoch", "value": to_epoch_midnight(from_date)}}

    # Fallback query if user only gave filters
    query_text = question or (f"events in {city}" if city else "city events")

    req = {
        "knowledgeBaseId": kb_id,
        "retrievalQuery": {"text": query_text},
        "retrievalConfiguration": {
            "vectorSearchConfiguration": { "numberOfResults": top_k }
        }
    }
    if filter_clause:
        req["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"] = filter_clause

    try:
        resp = agent_runtime.retrieve(**req)
    except Exception as e:
        print(f"[ERROR] retrieve: {e}")
        return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": json.dumps({"events": []})}

    results = resp.get("retrievalResults", [])
    events = []
    for r in results:
        content = r.get("content", {})
        metadata = r.get("metadata", {}) or {}
        text = content.get("text") or ""
        # Metadata from ingestion
        city_val  = metadata.get("city", city)
        date_iso  = metadata.get("event_date_iso")
        desc = ""
        name = ""
        if ": " in text:
            try:
                left, desc = text.split(": ", 1)
                # left looks like "EventName in City on YYYY-MM-DD"
                name = left
                if date_iso and f" on {date_iso}" in name:
                    name = name.replace(f" on {date_iso}", "")
                if city_val and f" in {city_val}" in name:
                    name = name.replace(f" in {city_val}", "")
                name = name.strip()
                desc = desc.strip()
            except Exception:
                name = text[:50]
                desc = text
        else:
            name = text[:50]
            desc = text

        events.append({
            "city": city_val,
            "event_date": date_iso,
            "event_name": name,
            "description": desc
        })

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"events": events})
    }
