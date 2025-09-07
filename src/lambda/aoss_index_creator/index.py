# Python 3.12
"""
Custom resource Lambda to create an OpenSearch Serverless vector index.

It is invoked by CloudFormation with event['RequestType'] in {'Create','Update','Delete'}.
On Create/Update, it PUTs an index with:
  - settings: index.knn=true (enables k-NN)
  - mappings:
      vector (knn_vector, dimension=1024, HNSW, cosine)
      text   (text)
      metadata (object)  # free-form metadata
"""
import json
import os
import urllib.request
import urllib.error
import time
from urllib.parse import urljoin

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

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
    req = urllib.request.Request(response_url, data=json.dumps(body).encode("utf-8"), method="PUT")
    req.add_header("Content-Type", "")
    urllib.request.urlopen(req)

def _signed_request(method, url, region, service, body=None, headers=None):
    session = boto3.session.Session()
    creds = session.get_credentials().get_frozen_credentials()
    request = AWSRequest(method=method, url=url, data=body, headers=headers or {})
    SigV4Auth(creds, service, region).add_auth(request)
    prepared = request.prepare()
    opener = urllib.request.build_opener()
    req = urllib.request.Request(url, data=prepared.body, method=method)
    for k, v in prepared.headers.items():
        req.add_header(k, v)
    return opener.open(req)

def _create_index(aoss_endpoint: str, index_name: str):
    """
    Creates a vector index with:
      knn_vector 'vector' (dimension 1024, HNSW, cosine)
      text 'text'
      object 'metadata'
    """
    # OpenSearch Serverless collection endpoint looks like:
    # https://<id>.<region>.aoss.amazonaws.com
    # Create index API: PUT /{index_name}
    url = urljoin(aoss_endpoint.rstrip("/") + "/", index_name)
    payload = {
        "settings": {
            "index": {
                "knn": True,
                # optional tuning could go here; keep minimal
            }
        },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": 1024,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "faiss"
                    }
                },
                "text": {"type": "text"},
                "metadata": {"type": "object", "enabled": True}
            }
        }
    }
    body = json.dumps(payload).encode("utf-8")
    resp = _signed_request(
        method="PUT",
        url=url,
        region=os.environ.get("AWS_REGION", "us-east-1"),
        service="aoss",
        body=body,
        headers={"Content-Type": "application/json"}
    )
    return resp.read()

def handler(event, context):
    try:
        req_type = event["RequestType"]
        props = event["ResourceProperties"]
        aoss_endpoint = props["CollectionEndpoint"]  # from CFN attribute
        index_name = props["IndexName"]

        if req_type == "Delete":
            # Leave index for diagnostics/cost negligible; report success.
            _respond(event, context, SUCCESS, data={"message": "No-op on delete"})
            return

        # Create (or idempotent re-create)
        try:
            _create_index(aoss_endpoint, index_name)
        except urllib.error.HTTPError as e:
            # If it already exists (HTTP 400/409), treat as success (idempotent)
            if e.code in (400, 409):
                _respond(event, context, SUCCESS, data={"message": "Index exists"})
                return
            raise

        _respond(event, context, SUCCESS, data={"message": "Index created"})
    except Exception as e:
        _respond(event, context, FAILED, reason=str(e))
