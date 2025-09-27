import os
import json
import math
import boto3
from datetime import datetime, timezone

s3 = boto3.client('s3')
bedrock_agent = boto3.client('bedrock-agent')  # direct ingestion API

MAX_PER_CALL = 10  # Bedrock KB docs per request

def to_epoch_midnight(date_iso: str) -> float:
    # date_iso: "YYYY-MM-DD"
    dt = datetime.strptime(date_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc, hour=0, minute=0, second=0, microsecond=0)
    return float(dt.timestamp())

def build_doc(evt):
    city = evt.get("city", "").strip()
    name = evt.get("event_name", "").strip()
    date_iso = evt.get("event_date", "").strip()
    desc = evt.get("description", "").strip()
    tags = evt.get("tags", None)

    epoch = to_epoch_midnight(date_iso) if date_iso else 0.0
    text = f"{name} in {city} on {date_iso}: {desc}".strip()

    attrs = [
        {"key": "city", "value": {"type": "string", "stringValue": city}},
        {"key": "event_date_iso", "value": {"type": "string", "stringValue": date_iso}},
        {"key": "event_epoch", "value": {"type": "number", "numberValue": epoch}}
    ]
    if isinstance(tags, list) and tags:
        # Store as multiple string attributes, or a string_list if supported
        attrs.append({"key": "tags", "value": {"type": "stringList", "stringListValue": tags}})

    return {
        "content": {
            "dataSourceType": "CUSTOM",
            "custom": {
                "customDocumentIdentifier": {"id": f"{city}|{date_iso}|{name}"},
                "inlineContent": {
                    "type": "text/plain",
                    "textContent": {"data": text}
                }
            }
        },
        "metadata": {
            "inlineAttributes": attrs
        }
    }

def handler(event, context):
    kb_id = os.environ["KNOWLEDGE_BASE_ID"]
    ds_id = os.environ["DATA_SOURCE_ID"]
    bucket = os.environ["EVENTS_S3_BUCKET"]
    key = os.environ["EVENTS_S3_KEY"]

    obj = s3.get_object(Bucket=bucket, Key=key)
    items = json.loads(obj["Body"].read())

    docs = [build_doc(evt) for evt in items]
    total = len(docs)
    print(f"[LOAD] {total} events from s3://{bucket}/{key}")

    # Batch calls
    ingested = 0
    for i in range(0, total, MAX_PER_CALL):
        batch = docs[i:i+MAX_PER_CALL]
        try:
            resp = bedrock_agent.ingest_knowledge_base_documents(
                knowledgeBaseId=kb_id,
                dataSourceId=ds_id,
                documents=batch
            )
            ingested += len(batch)
            print(f"[INGEST] {ingested}/{total} | resp={resp.get('documentDetails', [])}")
        except Exception as e:
            print(f"[ERROR] ingest batch starting at {i}: {e}")

    return {"status": "OK", "ingested": ingested, "total": total}
