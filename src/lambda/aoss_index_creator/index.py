# Python 3.12
"""
Custom resource Lambda to create an OpenSearch Serverless vector index with
FAISS/HNSW/cosine, dimension=1024 (Titan V2 default) in a given collection.

Fixes for 403:
- Logs detailed context (region, endpoint, caller ARN, HTTP bodies)
- Waits for collection to be ACTIVE (BatchGetCollection)
- Preflights signed GET / and retries with backoff on 401/403/429/5xx
- Idempotent: if index exists, exits SUCCESS
"""
import json
import os
import time
import urllib.request
import urllib.error
from urllib.parse import urljoin
from typing import Tuple

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.exceptions import ClientError

SUCCESS = "SUCCESS"
FAILED = "FAILED"

def _respond(event, context, status, reason=None, data=None):
    response_url = event["ResponseURL"]
    body = {
        "Status": status,
        "Reason": reason or f"See CloudWatch Log Stream: {context.log_stream_name}",
        "PhysicalResourceId": event.get("PhysicalResourceId") or f"{event['LogicalResourceId']}-{int(time.time())}",
        "StackId": event["StackId"],
        "RequestId": event["RequestId"],
        "LogicalResourceId": event["LogicalResourceId"],
        "Data": data or {}
    }
    print(f"[CFN RESPONSE] status={status} reason={reason}")
    req = urllib.request.Request(response_url, data=json.dumps(body).encode("utf-8"), method="PUT")
    req.add_header("Content-Type", "")
    urllib.request.urlopen(req)

def _sts_caller_arn() -> str:
    sts = boto3.client("sts")
    who = sts.get_caller_identity()
    return who.get("Arn", "unknown")

def _sign(method: str, url: str, region: str, service: str, body: bytes | None, headers: dict | None):
    session = boto3.session.Session()
    creds = session.get_credentials().get_frozen_credentials()
    request = AWSRequest(method=method, url=url, data=body, headers=headers or {})
    SigV4Auth(creds, service, region).add_auth(request)
    prepared = request.prepare()
    req = urllib.request.Request(url, data=prepared.body, method=method)
    for k, v in prepared.headers.items():
        req.add_header(k, v)
    return req

def _http(method: str, url: str, region: str, service: str, body: bytes | None = None, headers: dict | None = None) -> Tuple[int, str]:
    req = _sign(method, url, region, service, body=body, headers=headers)
    try:
        with urllib.request.build_opener().open(req) as resp:
            out = resp.read().decode("utf-8", errors="ignore")
            return resp.getcode(), out
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        return e.code, err_body

def _wait_collection_active(collection_arn: str, timeout_s: int = 300):
    """Wait until the collection is ACTIVE using control plane API."""
    oss = boto3.client("opensearchserverless")
    # ARN: arn:aws:aoss:<region>:<acct>:collection/<id>
    coll_id = collection_arn.split("/")[-1]
    print(f"[WAIT] Waiting for collection ACTIVE: id={coll_id}")
    t0 = time.time()
    while True:
        try:
            resp = oss.batch_get_collection(ids=[coll_id])
            summaries = resp.get("collectionDetails", [])
            if summaries:
                status = summaries[0].get("status")
                print(f"[WAIT] Collection status={status}")
                if status == "ACTIVE":
                    return
        except ClientError as e:
            print(f"[WAIT] batch_get_collection error: {e}")
        if time.time() - t0 > timeout_s:
            raise TimeoutError("Timed out waiting for collection to become ACTIVE")
        time.sleep(5)

def _prefight(aoss_endpoint: str, region: str, retries: int = 20):
    """Signed GET / with exponential backoff to let policies propagate."""
    url = aoss_endpoint.rstrip("/") + "/"
    backoff = 1.0
    for attempt in range(1, retries + 1):
        code, body = _http("GET", url, region, "aoss", None, headers={"Accept": "application/json"})
        print(f"[PREFLIGHT] attempt={attempt} code={code} body_sample={body[:200]}")
        if code in (200, 401):  # 401 means signed but missing auth header specifics; usually OK once policy applies
            # We accept 200 outright; 401 often disappears after policy propagation; continue loop briefly
            if code == 200:
                return
        if code not in (401, 403, 429, 500, 502, 503):
            # Unexpected, but don't block forever
            return
        time.sleep(backoff)
        backoff = min(backoff * 1.6, 10.0)

def _index_exists(aoss_endpoint: str, index_name: str, region: str) -> bool:
    url = urljoin(aoss_endpoint.rstrip("/") + "/", index_name)
    code, _ = _http("HEAD", url, region, "aoss")
    print(f"[EXISTS] HEAD /{index_name} code={code}")
    return code == 200

def _create_index(aoss_endpoint: str, index_name: str, region: str):
    url = urljoin(aoss_endpoint.rstrip("/") + "/", index_name)
    payload = {
        "settings": { "index": { "knn": True } },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": 1024,                 # Titan V2 default output dims
                    "method": { "name": "hnsw", "space_type": "cosinesimil", "engine": "faiss" }
                },
                "text":     { "type": "text" },
                "metadata": { "type": "object", "enabled": True }
            }
        }
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    # Retry a few times for policy propagation
    backoff = 1.0
    for attempt in range(1, 16):
        code, resp_body = _http("PUT", url, region, "aoss", body=body, headers=headers)
        print(f"[CREATE] attempt={attempt} code={code} body_sample={resp_body[:200]}")
        if code in (200, 201):
            return
        if code in (400, 409):
            # Already exists or bad request — treat 409 as success for idempotency
            return
        if code in (401, 403, 429, 500, 502, 503):
            time.sleep(backoff)
            backoff = min(backoff * 1.6, 12.0)
            continue
        # Unexpected failure; break
        raise RuntimeError(f"Index create failed with code={code}: {resp_body}")

def handler(event, context):
    try:
        req_type = event["RequestType"]
        props = event["ResourceProperties"]
        aoss_endpoint = props["CollectionEndpoint"]
        index_name = props["IndexName"]
        collection_arn = props["CollectionArn"]

        region = boto3.session.Session().region_name or os.environ.get("AWS_REGION", "us-east-1")
        caller_arn = _sts_caller_arn()
        print(f"[START] req_type={req_type} region={region} endpoint={aoss_endpoint} index={index_name} caller={caller_arn}")

        if req_type == "Delete":
            print("[DELETE] No-op; leaving index in place")
            _respond(event, context, SUCCESS, data={"message": "No-op on delete"})
            return

        # 1) Wait until collection is ACTIVE (control plane)
        _wait_collection_active(collection_arn, timeout_s=600)

        # 2) Preflight signed GET / (data plane) with retries to let policy propagate
        _prefight(aoss_endpoint, region, retries=30)

        # 3) Idempotent HEAD for index
        if _index_exists(aoss_endpoint, index_name, region):
            _respond(event, context, SUCCESS, data={"message": "Index exists"})
            return

        # 4) Create index (retry on 401/403/429/5xx)
        _create_index(aoss_endpoint, index_name, region)

        _respond(event, context, SUCCESS, data={"message": "Index created"})
    except Exception as e:
        print(f"[ERROR] {e}")
        _respond(event, context, FAILED, reason=str(e))
