import json
import time
import os
import traceback
import urllib3
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

http = urllib3.PoolManager(num_pools=10, maxsize=10, retries=False, timeout=urllib3.util.Timeout(total=30.0))
aoss = boto3.client('opensearchserverless')
sts = boto3.client('sts')

def _cf_response(event, context, status, reason=None):
    url = event.get('ResponseURL')
    if not url:
        # CFN did not receive status
        print("[WARN] No ResponseURL in event.")
        return
    body = {
        'Status': status,
        'Reason': reason or f"See CloudWatch Log Stream: {getattr(context,'log_stream_name','')}",
        'PhysicalResourceId': event.get('PhysicalResourceId') or f"{event['LogicalResourceId']}-{int(time.time())}",
        'StackId': event['StackId'],
        'RequestId': event['RequestId'],
        'LogicalResourceId': event['LogicalResourceId'],
        'NoEcho': False,
        'Data': data or {}
    }
    print(f"[CFN RESPONSE] status={status} reason={reason}")  # CHANGED: added logging
    try:
        http.request('PUT', url, body=json.dumps(body).encode('utf-8'),
                 headers={'Content-Type': 'application/json'})
    except Exception as e:  # CHANGED: new exception handling
        print(f"[ERROR] Sending CFN response failed: {e}")

def _sign_and_send(region, method, url, body=None, headers=None):
    session = boto3.session.Session()
    creds = session.get_credentials()
    if creds is None:  # CHANGED: explicit guard
        raise RuntimeError("No AWS credentials available to sign AOSS request")  # CHANGED: new error
    frozen = creds.get_frozen_credentials()
    req = AWSRequest(method=method, url=url, data=body or b'',
                     headers=headers or {'Content-Type': 'application/json'})
    SigV4Auth(frozen, 'aoss', region).add_auth(req)
    signed_headers = dict(req.headers.items())
    resp = http.request(method, url, body=body, headers=signed_headers)
    return resp


# -------------- AOSS helpers --------------

def _get_collection_endpoint(collection_name):
    resp = aoss.batch_get_collection(names=[collection_name])
    details = resp.get('collectionDetails', [])
    if not details:
        raise RuntimeError(f"Collection '{collection_name}' not found")
    return details[0]['collectionEndpoint'], details[0]['status']

def _wait_collection_active(collection_name, region, max_wait_s=300, poll_s=5):
    waited = 0
    while waited <= max_wait_s:
        _, status = _get_collection_endpoint(collection_name)
        print(f"[WAIT] Collection '{collection_name}' status={status}")
        if status == 'ACTIVE':
            return
        time.sleep(poll_s)
        waited += poll_s
    raise TimeoutError(f"Collection '{collection_name}' did not become ACTIVE within {max_wait_s}s")

def _index_exists(base_endpoint, region, index_name):
    head_url = f"{base_endpoint}/{index_name}"
    r = _sign_and_send(region, 'HEAD', head_url)
    print(f"[EXISTS] HEAD /{index_name} code={r.status}")
    return r.status == 200

def _create_index(base_endpoint, region, index_name):                                # CHANGED: mapping creation moved into its own function
    mapping = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100
            }
        },
        "mappings": {
            "properties": {
                # Vector field sized for Titan Text Embeddings v2 (1024 dims)       # CHANGED: clarifying comment
                "vector": {
                    "type": "knn_vector",
                    "dimension": 1024,
                    "space_type": "cosinesimil",
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "parameters": {"m": 16, "ef_construction": 128}
                    }
                },
                "text": {"type": "text"},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "keyword"},
                        "event_date_iso": {"type": "keyword"},
                        "event_epoch": {"type": "double"},
                        "tags": {"type": "keyword"}
                    }
                }
            }
        }
    }
    put_url = f"{base_endpoint}/{index_name}"
    pr = _sign_and_send(region, 'PUT', put_url,
                        body=json.dumps(mapping).encode('utf-8'),
                        headers={'Content-Type': 'application/json'})
    sample = (pr.data[:300].decode('utf-8', errors='ignore') if pr.data else '')     # CHANGED: decode body sample for readable logs
    print(f"[CREATE] PUT /{index_name} code={pr.status} body_sample={sample}")       # CHANGED: new log format
    if pr.status not in (200, 201):
        raise RuntimeError(f"Create index failed: HTTP {pr.status} {sample}")        # CHANGED: raise instead of CFN response here

def _stabilize_index(base_endpoint, region, index_name, max_wait_s=300, backoff_start_s=2, backoff_cap_s=20):  # CHANGED: new helper for robust stabilization
    """
    Wait until:
      - HEAD /{index} == 200
      - GET /{index}/_mapping == 200
      - GET /{index}/_search?size=0 == 200
    Backoff starts at 2s, doubles up to 20s, total <= ~5 minutes.
    """
    deadline = time.time() + max_wait_s
    delay = backoff_start_s
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        # HEAD
        head_url = f"{base_endpoint}/{index_name}"
        hr = _sign_and_send(region, 'HEAD', head_url)
        print(f"[STABILIZE] attempt={attempt} HEAD code={hr.status}")
        if hr.status != 200:
            time.sleep(delay)
            delay = min(delay * 2, backoff_cap_s)
            continue

        # _mapping
        map_url = f"{base_endpoint}/{index_name}/_mapping"
        mr = _sign_and_send(region, 'GET', map_url)
        print(f"[STABILIZE] attempt={attempt} _mapping code={mr.status}")
        if mr.status != 200:
            time.sleep(delay)
            delay = min(delay * 2, backoff_cap_s)
            continue

        # _search?size=0
        search_url = f"{base_endpoint}/{index_name}/_search?size=0"
        sr = _sign_and_send(region, 'GET', search_url)
        print(f"[STABILIZE] attempt={attempt} _search code={sr.status}")
        if sr.status == 200:
            print("[READY] Index is queryable")
            return

        time.sleep(delay)
        delay = min(delay * 2, backoff_cap_s)

    raise TimeoutError(f"Index '{index_name}' did not become queryable within {max_wait_s}s")

# -------------- Lambda entrypoint --------------

def handler(event, context):
    """
    Expected custom resource event properties:
      - CollectionName
      - IndexName
    """
    print(f"[EVENT] {json.dumps(event)}")
    req_type = event.get('RequestType', 'Create')
    props = event.get('ResourceProperties', {}) or {}
    collection_name = props.get('CollectionName')
    index_name = props.get('IndexName')
    region = os.environ.get('AWS_REGION', 'us-east-1')

    physical_id = f"{collection_name}:{index_name}"

    try:
        if req_type == 'Delete':
            # Leave index intact on stack delete (safer in dev); signal success.
            _cfn_response(event, context, "SUCCESS", data={"action": "delete_noop"}, physical_id=physical_id)
            return

        # Create / Update path
        if not collection_name or not index_name:
            raise ValueError("Missing required properties: CollectionName and IndexName")

        # Wait for collection ACTIVE and get its data-plane endpoint
        _wait_collection_active(collection_name, region)
        endpoint, _ = _get_collection_endpoint(collection_name)
        base = endpoint.rstrip('/')

        # If index exists, just stabilize (it may still be warming)
        if _index_exists(base, region, index_name):
            try:
                _stabilize_index(base, region, index_name)
            except Exception as e:
                # If stabilize fails, still surface the reason
                raise
            _cfn_response(event, context, "SUCCESS",
                          data={"action": "exists", "endpoint": endpoint, "index": index_name},
                          physical_id=physical_id)
            return

        # Create the index
        _create_index(base, region, index_name)
        # Stabilize (HEAD + _mapping + _search)
        _stabilize_index(base, region, index_name)

        _cfn_response(event, context, "SUCCESS",
                      data={"action": "created", "endpoint": endpoint, "index": index_name},
                      physical_id=physical_id)
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
        print(f"[ERROR] {reason}")
        traceback.print_exc()
        _cfn_response(event, context, "FAILED", reason=reason, physical_id=physical_id)
