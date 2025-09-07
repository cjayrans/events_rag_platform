Prereqs

AWS account 769265885190, Region us-east-1

Dataset is already in S3: s3://sagemaker-us-east-1-769265885190/datasets/city_events/eventsDB_data.json

You have access to AWS Console and GitHub

A) Create the new GitHub repo (GUI)

Go to github.com → top-right + → New repository.

Repository name: events_rag_platform → Public or Private (your choice) → Create repository.

On the new repo page, click Add file → Upload files.

Upload the entire local folder events_rag_platform/ with this structure:

infra/cloudformation/events_rag_platform.yaml

src/lambda/retrieval/index.py

README.md, .gitignore, __init__.py

Click Commit changes.

If you prefer from PyCharm: VCS → Enable Version Control → Git → Share Project on GitHub → Pick name events_rag_platform → Share.

B) Package the Lambda code (no CLI; Finder only)

On your Mac, open Finder and navigate to your local repo folder.

Go to events_rag_platform/src/lambda/retrieval/.

Select index.py and __init__.py → right-click → Compress 2 items.

Rename the generated zip to retrieval.zip.

(No external libraries are needed; the function uses only boto3 + AWS Bedrock KB API.)

C) Upload the zip to S3 (GUI)

In the AWS Console, switch to N. Virginia (us-east-1).

Open S3 → Create bucket.

Bucket name: events-rag-artifacts-769265885190-us-east-1 (exact spelling, all lowercase).

Leave defaults (Block Public Access on). Click Create bucket.

Click the bucket → Create folder → Name: lambda → Create folder.

Click into the lambda/ folder → Upload → Add files → select retrieval.zip → Upload.

After upload, click the object and note its Key (should be lambda/retrieval.zip).

D) Deploy the infrastructure (one CloudFormation template; GUI)

Open CloudFormation → Create stack → With new resources (standard).

Specify template: Choose Upload a template file → Choose file → select events_rag_platform.yaml from your local repo → Next.

Stack name: CityEvents-RAG.

Parameters:

KnowledgeBaseName: CityEventsKB (default ok)

DataSourceName: CityEventsData (default ok)

AOSSCollectionName: city-events-collection (default ok)

AOSSIndexName: city-events-index (default ok)

LambdaFunctionName: CityEventsRetrievalFunction (default ok)

LambdaCodeS3Bucket: events-rag-artifacts-769265885190-us-east-1

LambdaCodeS3Key: lambda/retrieval.zip

NumResults: 5 (default ok)

Next → keep defaults → Next → at bottom, acknowledge IAM creation → Create stack.

Wait until status is CREATE_COMPLETE.

E) Sync your dataset into the Knowledge Base (GUI)

Go to Amazon Bedrock → left nav Knowledge bases.

Click your KB named CityEventsKB (from the stack).

In Data sources tab, you should see CityEventsData pointing to your S3 JSON.

Click Sync → confirm.

Bedrock KB will chunk your JSON, generate Amazon Titan Text Embeddings V2 (default 1024-dim optimized for RAG) and index into OpenSearch Serverless automatically.

When Sync completes, status shows Active and documents count > 0.

F) Get the Retrieval Function URL

Open Lambda → click CityEventsRetrievalFunction.

On Configuration → Function URL, copy the URL (looks like https://xxxx.lambda-url.us-east-1.on.aws).

Auth is set to AWS_IAM. Calls must be SigV4-signed (your chatbot Lambda will have IAM and will invoke it server-to-server; browsers will be blocked by design).

G) Quick validation (no CLI)

In the same Lambda page → Test → Configure test event.

Event name: miami

Event JSON:

{ "city": "Miami" }


Save, then Test.

Check the Execution result panel → Response → it should be:

{ "events": "Art Basel on 2024-12-05: An international art fair.\n..." }


(Your actual text depends on your dataset contents; you’ll see up to 3 lines.)

If you want to see the raw KB hits, open CloudWatch Logs (link from Lambda page) for this function. Minimal logs only.