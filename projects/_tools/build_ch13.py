"""Generate the Chapter 13 notebook: the AWS services from the lecture, simulated offline with moto."""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "ch13_cloud_fundamentals", "aws_services_offline.ipynb")


def md(s): return nbf.v4.new_markdown_cell(s.strip("\n"))
def py(s): return nbf.v4.new_code_cell(s.strip("\n"))


cells = [
md("""
# Chapter 13 · Cloud fundamentals: the AWS services, hands-on and offline

Companion to `notes/13_cloud_fundamentals_for_mlops.html` (video 03:09:31 – 03:26:39).

In the video the instructor tours the AWS and GCP consoles. This notebook uses the **same AWS services from Python (`boto3`)**, but against **[moto](https://github.com/getmoto/moto)**, a local fake of AWS:

* **No AWS account, no credentials, no cost.** Nothing leaves your laptop.
* The code is exactly what you'd run against real AWS. Remove the mock (and configure real credentials) and it works the same way.

| Section | Service | Why it matters in the course |
|---|---|---|
| 1 | Regions | pick the one closest to you and your users |
| 2 | **S3** | store data, models, DVC/MLflow artifacts |
| 3 | **EC2** + security groups | the Linux server from Chapter 05; deployment target |
| 4 | **ECR** | private registry for Docker images (CI/CD chapters) |
| 5 | **IAM** | least-privilege users instead of the root account |
"""),
py("""
import os, json, hashlib, pathlib, tempfile
import pandas as pd

# Fake credentials, and make sure a real ~/.aws config is NEVER used by this notebook
os.environ.update({
    "AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing", "AWS_SESSION_TOKEN": "testing",
    "AWS_DEFAULT_REGION": "ap-south-1",
    "AWS_CONFIG_FILE": os.devnull, "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
})

import boto3
from moto import mock_aws
from moto.core import set_initial_no_auth_action_count

REGION = "ap-south-1"          # Mumbai: the instructor's suggestion for learners in India
mock = mock_aws()
mock.start()                    # every boto3 call below now goes to the local fake AWS
print("boto3", boto3.__version__, "| mocked AWS started | region:", REGION)
"""),
md("""
## 1 · Regions  (video 03:17)

AWS runs data centres in many **regions**. The console shows the current one at the top right (the instructor was in `us-east-1`, N. Virginia).
Pick the region **closest to your users** for lower latency. Resources live **inside a region**: a bucket or instance created in Mumbai won't show up while the console is set to N. Virginia.
The region list below comes from the botocore data bundled with boto3, so no network is needed:
"""),
py("""
import botocore.session
endpoints = botocore.session.get_session().get_data("endpoints")
aws = next(p for p in endpoints["partitions"] if p["partition"] == "aws")
regions = pd.DataFrame(
    [(code, info.get("description", "")) for code, info in aws["regions"].items()],
    columns=["region code", "location"],
).sort_values("region code").reset_index(drop=True)
mentioned = {"us-east-1": "instructor's region", "ap-south-1": "suggested for India",
             "ap-northeast-3": "Osaka (shown in video)", "eu-west-1": "an option for Europe"}
regions["in the video"] = regions["region code"].map(mentioned).fillna("")
print(len(regions), "commercial AWS regions")
regions[regions["in the video"] != ""]
"""),
md("""
## 2 · Same service, different name on each cloud  (video 03:13)

The instructor's point: every cloud has the same building blocks with different names and UIs. **Master one cloud and the others become easy.**
"""),
py("""
mapping = pd.DataFrame([
    ("Virtual machine",          "EC2",                "Compute Engine (VM instances)", "Virtual Machines"),
    ("Object storage (buckets)", "S3",                 "Cloud Storage",                 "Blob Storage"),
    ("Container registry",       "ECR",                "Artifact Registry",             "Container Registry (ACR)"),
    ("Identity & permissions",   "IAM",                "Cloud IAM",                     "Entra ID + RBAC"),
    ("Managed ML platform",      "SageMaker",          "Vertex AI",                     "Azure Machine Learning"),
    ("CI/CD pipeline",           "CodePipeline",       "Cloud Build",                   "Azure Pipelines"),
    ("Managed app hosting",      "Elastic Beanstalk",  "App Engine",                    "App Service"),
    ("Monitoring & logs",        "CloudWatch",         "Cloud Monitoring / Logging",    "Azure Monitor"),
], columns=["capability", "AWS", "GCP", "Azure"])
mapping
"""),
md("""
## 3 · S3: store and retrieve any file  (video 03:20)

Upload the model and metrics from the Chapter 12 pipeline, list them, download one back, and check it's byte-for-byte identical.
"""),
py("""
s3 = boto3.client("s3", region_name=REGION)
BUCKET = "mlops-course-artifacts-demo"          # bucket names are globally unique on real AWS
s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION})

ch12 = pathlib.Path("../ch12_dvc_pipeline")
files = {"models/emotion/model.pkl": ch12 / "model.pkl", "models/emotion/metrics.json": ch12 / "metrics.json"}
for key, path in files.items():
    body = path.read_bytes() if path.exists() else b"placeholder (run the ch12 pipeline first)"
    s3.put_object(Bucket=BUCKET, Key=key, Body=body)

listing = s3.list_objects_v2(Bucket=BUCKET, Prefix="models/")
pd.DataFrame([(o["Key"], o["Size"]) for o in listing["Contents"]], columns=["key", "bytes"])
"""),
py("""
with tempfile.TemporaryDirectory() as tmp:
    local = pathlib.Path(tmp) / "model.pkl"
    s3.download_file(BUCKET, "models/emotion/model.pkl", str(local))
    same = hashlib.md5(local.read_bytes()).hexdigest() == hashlib.md5(files["models/emotion/model.pkl"].read_bytes()).hexdigest() \\
        if files["models/emotion/model.pkl"].exists() else "n/a"
print("downloaded copy identical to the original:", same)

print(json.loads(s3.get_object(Bucket=BUCKET, Key="models/emotion/metrics.json")["Body"].read()))

# a time-limited link you could hand to someone without giving them AWS access (beyond the video)
url = s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": "models/emotion/metrics.json"}, ExpiresIn=900)
print(url.split("?")[0], "?…(signed, expires in 15 min)")
"""),
md("""
## 4 · EC2: the virtual machine from Chapter 05, driven by code

Create the same kind of setup as the console demo: a security group with ports 22/80/443, then a **t2.micro** instance.
Then walk it through the lifecycle: running → stopped → terminated.
"""),
py("""
ec2 = boto3.client("ec2", region_name=REGION)

sg = ec2.create_security_group(GroupName="web-and-ssh", Description="SSH + HTTP + HTTPS")
ec2.authorize_security_group_ingress(GroupId=sg["GroupId"], IpPermissions=[
    {"IpProtocol": "tcp", "FromPort": p, "ToPort": p, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]} for p in (22, 80, 443)
])
key = ec2.create_key_pair(KeyName="linux-test")
image_id = ec2.describe_images(Owners=["amazon"])["Images"][0]["ImageId"]   # any AMI from the fake catalogue

inst = ec2.run_instances(ImageId=image_id, InstanceType="t2.micro", MinCount=1, MaxCount=1,
                         KeyName="linux-test", SecurityGroupIds=[sg["GroupId"]])["Instances"][0]
iid = inst["InstanceId"]
print("launched", iid, inst["InstanceType"], "| key pair:", key["KeyName"], "| private key starts:", key["KeyMaterial"][:27], "…")
rules = ec2.describe_security_groups(GroupIds=[sg["GroupId"]])["SecurityGroups"][0]["IpPermissions"]
print("open ports:", sorted(r["FromPort"] for r in rules))
"""),
py("""
def state():
    return ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["State"]["Name"]

timeline = [("after run_instances", state())]
ec2.stop_instances(InstanceIds=[iid]);      timeline.append(("after stop_instances", state()))
ec2.start_instances(InstanceIds=[iid]);     timeline.append(("after start_instances", state()))
ec2.terminate_instances(InstanceIds=[iid]); timeline.append(("after terminate_instances", state()))
pd.DataFrame(timeline, columns=["action", "state"])
"""),
md("""
Real AWS passes through `pending`, `stopping` and `shutting-down` in between. The billing rule from Chapter 05 still applies: **running** means you pay for compute, **stopped** means you pay for the disk only, **terminated** means you pay nothing.
"""),
md("""
## 5 · ECR: a private home for Docker images  (video 03:20)

Docker Hub is where public images live (the instructor shows the PyTorch image). **ECR** is AWS's registry for your *own* images, with access controlled by IAM.
The CI/CD chapters build an image and push it here.
"""),
py("""
ecr = boto3.client("ecr", region_name=REGION)
repo = ecr.create_repository(repositoryName="emotion-api")["repository"]
print("repository URI:", repo["repositoryUri"])
print("\\nwhat the push looks like with real AWS + Docker:")
print(f"  aws ecr get-login-password --region {REGION} | docker login --username AWS --password-stdin {repo['repositoryUri'].split('/')[0]}")
print(f"  docker tag emotion-api:latest {repo['repositoryUri']}:v1")
print(f"  docker push {repo['repositoryUri']}:v1")
"""),
md("""
## 6 · IAM: least privilege instead of the root account  (video 03:25)

The instructor's warning: if an application uses **admin/root** access keys, it can touch *every* service, and a mistake or a leak can run up a huge bill.
Instead, create a **user with only the permissions it needs**. For example, a developer who only needs S3.

Here moto **enforces** IAM. The first few calls (setup) run as admin, and every call after that is checked against the new user's policy.
"""),
py("""
S3_ONLY_POLICY = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Action": ["s3:ListBucket", "s3:GetObject", "s3:PutObject"],
        "Resource": ["arn:aws:s3:::team-data-bucket", "arn:aws:s3:::team-data-bucket/*"],
    }],
}
print(json.dumps(S3_ONLY_POLICY, indent=2))
"""),
py("""
@set_initial_no_auth_action_count(4)      # the first 4 AWS calls below are the admin setup
def least_privilege_demo():
    admin_iam = boto3.client("iam")
    admin_s3 = boto3.client("s3", region_name=REGION)
    admin_s3.create_bucket(Bucket="team-data-bucket", CreateBucketConfiguration={"LocationConstraint": REGION})  # 1
    admin_iam.create_user(UserName="dev-s3-only")                                                                # 2
    admin_iam.put_user_policy(UserName="dev-s3-only", PolicyName="s3-only",
                              PolicyDocument=json.dumps(S3_ONLY_POLICY))                                         # 3
    creds = admin_iam.create_access_key(UserName="dev-s3-only")["AccessKey"]                                     # 4

    # from here on every call is checked against dev-s3-only's policy
    dev = boto3.Session(aws_access_key_id=creds["AccessKeyId"],
                        aws_secret_access_key=creds["SecretAccessKey"], region_name=REGION)
    attempts = [
        ("upload to team-data-bucket", lambda: dev.client("s3").put_object(Bucket="team-data-bucket", Key="train.csv", Body=b"x,y")),
        ("list team-data-bucket",      lambda: dev.client("s3").list_objects_v2(Bucket="team-data-bucket")),
        ("create a new bucket",        lambda: dev.client("s3").create_bucket(Bucket="another-bucket", CreateBucketConfiguration={"LocationConstraint": REGION})),
        ("list EC2 instances",         lambda: dev.client("ec2").describe_instances()),
        ("create an IAM user",         lambda: dev.client("iam").create_user(UserName="sneaky-admin")),
    ]
    rows = []
    for name, call in attempts:
        try:
            call(); rows.append((name, "✅ allowed"))
        except Exception as e:
            rows.append((name, "⛔ " + (e.response["Error"]["Code"] if hasattr(e, "response") else type(e).__name__)))
    return pd.DataFrame(rows, columns=["action by dev-s3-only", "result"]), creds["AccessKeyId"]

result, key_id = least_privilege_demo()
print("access key id issued to the developer:", key_id[:4] + "…")
result
"""),
md("""
Only the actions in the policy work. Everything else, including launching machines or creating new admins, is denied. That's the **principle of least privilege**.

**Doing it for real** (after creating an IAM user and an access key in the console):

```bash
pip install awscli
aws configure                  # paste the key id + secret, pick a region such as ap-south-1
aws sts get-caller-identity    # which user am I?
aws s3 ls                      # list buckets
aws s3 cp model.pkl s3://<bucket>/models/
```

Never commit access keys to Git. On EC2, attach an **IAM role** to the instance instead of copying keys onto it.
"""),
py("""
mock.stop()
print("mocked AWS stopped; nothing was created in a real account")
"""),
]

book = nbf.v4.new_notebook()
book.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3 (MLOps .venv)", "language": "python"},
                 "language_info": {"name": "python"}}
book.cells = cells
os.makedirs(os.path.dirname(OUT), exist_ok=True)
nbf.write(book, OUT)
print("wrote", OUT)
