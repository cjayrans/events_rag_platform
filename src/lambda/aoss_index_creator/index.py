"""
Custom resource Lambda to create an OpenSearch Serverless vector index.

Invoked by CloudFormation with event['RequestType'] in {'Create','Update','Delete'}.

Creates index mapping:
  - settings.index.knn = true
  - properties:
      vector:   knn_vector, dimension=1024, method HNSW, space_type cosine (engine faiss)
      text:     text
      metadata: object (enabled)
"""
import json
import os
import time
import urllib.request
import urllib.error
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
    req = urllib.request.Request(url, data=prepared.body, method=method)
    for k, v in prepared.headers.items():
        req.add_header(k, v)
    return urllib.request.build_opener().open(req)

def _create_index(aoss_endpoint: str, index_name: str, region: str):
    """
    PUT /{index_name}
    """
    url = urljoin(aoss_endpoint.rstrip("/") + "/", index_name)
    payload = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": 1024,                 # Titan V2 default dims (match index)
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
    # NOTE: When signing to AOSS, the service name must be 'aoss' and SigV4 rules apply.
    return _signed_request(
        method="PUT",
        url=url,
        region=region,
        service="aoss",
        body=body,
        headers={"Content-Type": "application/json"}
    ).read()

def handler(event, context):
    try:
        req_type = event["RequestType"]
        props = event["ResourceProperties"]
        aoss_endpoint = props["CollectionEndpoint"]
        index_name = props["IndexName"]

        # Use runtime-provided region (don’t set any reserved env vars)
        region = boto3.session.Session().region_name or os.environ.get("AWS_REGION", "us-east-1")

        if req_type == "Delete":
            # Leave index intact (idempotent deletions). Report success.
            _respond(event, context, SUCCESS, data={"message": "No-op on delete"})
            return

        try:
            _create_index(aoss_endpoint, index_name, region)
        except urllib.error.HTTPError as e:
            # If already exists (400/409), treat as success for idempotency
            if e.code in (400, 409):
                _respond(event, context, SUCCESS, data={"message": "Index exists"})
                return
            raise

        _respond(event, context, SUCCESS, data={"message": "Index created"})
    except Exception as e:
        _respond(event, context, FAILED, reason=str(e))
