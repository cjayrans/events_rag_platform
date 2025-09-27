import json
import time
import os
import urllib3
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

http = urllib3.PoolManager()
aoss = boto3.client('opensearchserverless')
sts = boto3.client('sts')

def _cf_response(event, context, status, reason=None):
    url = event.get('ResponseURL')
    if not url:
        return
    body = {
        'Status': status,
        'Reason': reason or f"See CloudWatch Log Stream: {getattr(context,'log_stream_name','')}",
        'PhysicalResourceId': event.get('PhysicalResourceId') or f"{event['LogicalResourceId']}-{int(time.time())}",
        'StackId': event['StackId'],
        'RequestId': event['RequestId'],
        'LogicalResourceId': event['LogicalResourceId'],
        'Data': {}
    }
    http.request('PUT', url, body=json.dumps(body).encode('utf-8'),
                 headers={'Content-Type': 'application/json'})

def _sign_and_send(region, method, url, body=None, headers=None):
    session = boto3.session.Session()
    creds = session.get_credentials().get_frozen_credentials()
    req = AWSRequest(method=method, url=url, data=body or b'',
                     headers=headers or {'Content-Type': 'application/json'})
    SigV4Auth(creds, 'aoss', region).add_auth(req)
    rheaders = dict(req.headers.items())
    resp = http.request(method, url, body=body, headers=rheaders)
    return resp

def handler(event, context):
    print(f"[EVENT] {json.dumps(event)}")
    try:
        req_type = event.get('RequestType')
        props = event.get('ResourceProperties', {})
        collection_name = props['CollectionName']
        index_name = props['IndexName']
        region = os.environ.get('AWS_REGION', 'us-east-1')

        # Resolve collection endpoint (data plane)
        col = aoss.batch_get_collection(names=[collection_name])
        items = col.get('collectionDetails', [])
        if not items:
            _cf_response(event, context, "FAILED", reason=f"Collection {collection_name} not found")
            return
        endpoint = items[0]['collectionEndpoint']  # e.g., https://abc123.us-east-1.aoss.amazonaws.com
        base = endpoint.rstrip('/')

        if req_type == 'Delete':
            # For safety in dev, we leave the index in place (no-op)
            _cf_response(event, context, "SUCCESS")
            return

        # Preflight: check if index exists
        head_url = f"{base}/{index_name}"
        r = _sign_and_send(region, 'HEAD', head_url)
        if r.status == 200:
            print(f"[INFO] Index {index_name} already exists")
            _cf_response(event, context, "SUCCESS")
            return

        # Create index with mapping
        mapping = {
          "settings": {
            "index": {
              "knn": True,
              "knn.algo_param.ef_search": 100
            }
          },
          "mappings": {
            "properties": {
              "vector": {
                "type": "knn_vector",
                "dimension": 1024,
                "space_type": "cosinesimil",
                "method": {
                  "name": "hnsw",
                  "engine": "faiss",
                  "parameters": { "m": 16, "ef_construction": 128 }
                }
              },
              "text": { "type": "text" },
              "metadata": {
                "type": "object",
                "properties": {
                  "city": { "type": "keyword" },
                  "event_date_iso": { "type": "keyword" },
                  "event_epoch": { "type": "double" },
                  "tags": { "type": "keyword" }
                }
              }
            }
          }
        }
        put_url = f"{base}/{index_name}"
        pr = _sign_and_send(region, 'PUT', put_url,
                            body=json.dumps(mapping).encode('utf-8'),
                            headers={'Content-Type': 'application/json'})
        print(f"[CREATE] code={pr.status} body={pr.data[:300]}")
        if pr.status not in (200, 201):
            _cf_response(event, context, "FAILED", reason=f"Create index failed: {pr.status}")
            return

        # Stabilize: wait until mapping readable
        for attempt in range(30):
            time.sleep(5)
            map_url = f"{base}/{index_name}/_mapping"
            mr = _sign_and_send(region, 'GET', map_url)
            if mr.status == 200:
                print("[READY] Mapping visible")
                _cf_response(event, context, "SUCCESS")
                return
            print(f"[WAIT] attempt={attempt+1} status={mr.status}")
        _cf_response(event, context, "FAILED", reason="Index did not stabilize")
    except Exception as e:
        print(f"[ERROR] {e}")
        _cf_response(event, context, "FAILED", reason=str(e))
