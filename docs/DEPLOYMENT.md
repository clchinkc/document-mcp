# Deployment Guide

Deploy Document MCP to Google Cloud Run with minimal cost.

## Current Configuration

- **Project ID:** document-mcp-54749
- **Region:** asia-east1
- **Service Name:** document-mcp
- **Storage:** Firestore (free tier, scales to zero)
- **Authentication:** Google OAuth 2.1

## Quick Deploy (One Command)

Push to `main` branch triggers automatic deployment via GitHub Actions:

```bash
git push origin main
```

Or manually trigger from GitHub Actions → "Deploy to Cloud Run" → "Run workflow".

## Manual Deployment

```bash
# Set project
export PROJECT_ID="document-mcp-54749"
export REGION="asia-east1"
export SERVICE_NAME="document-mcp"

# Deploy from source
gcloud run deploy $SERVICE_NAME \
  --source . \
  --region $REGION \
  --project $PROJECT_ID \
  --allow-unauthenticated \
  --platform managed \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 10 \
  --port 8080 \
  --set-env-vars "DOCUMENTS_STORAGE_PATH=/data/documents_storage,LOG_LEVEL=info,SERVER_URL=https://document-mcp-451560119112.asia-east1.run.app" \
  --set-secrets "GOOGLE_OAUTH_CLIENT_ID=GOOGLE_OAUTH_CLIENT_ID:latest,GOOGLE_OAUTH_CLIENT_SECRET=GOOGLE_OAUTH_CLIENT_SECRET:latest"
```

## Full Infrastructure Setup (From Scratch)

If starting fresh or in a new project:

### 1. Enable Required APIs

```bash
export PROJECT_ID="document-mcp-54749"

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  firestore.googleapis.com \
  --project=$PROJECT_ID
```

### 2. Create Firestore Database

```bash
gcloud firestore databases create \
  --project=$PROJECT_ID \
  --location=asia-east1 \
  --type=firestore-native
```

### 3. Create Secrets (OAuth Credentials)

Get OAuth credentials from Google Cloud Console → APIs & Services → Credentials → Create OAuth 2.0 Client ID.

```bash
# Create secrets
echo -n "YOUR_CLIENT_ID" | gcloud secrets create GOOGLE_OAUTH_CLIENT_ID \
  --data-file=- --project=$PROJECT_ID

echo -n "YOUR_CLIENT_SECRET" | gcloud secrets create GOOGLE_OAUTH_CLIENT_SECRET \
  --data-file=- --project=$PROJECT_ID
```

### 4. Create Service Account for GitHub Actions

```bash
# Create service account
gcloud iam service-accounts create github-actions-deployer \
  --display-name="GitHub Actions Cloud Run Deployer" \
  --project=$PROJECT_ID

# Grant permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-deployer@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-deployer@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-deployer@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-deployer@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudbuild.builds.builder"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-deployer@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-deployer@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# Create key for GitHub Actions
gcloud iam service-accounts keys create github-actions-key.json \
  --iam-account=github-actions-deployer@$PROJECT_ID.iam.gserviceaccount.com

# Add to GitHub Secrets as GCP_SA_KEY
cat github-actions-key.json
rm github-actions-key.json  # Delete after adding to GitHub
```

### 5. Deploy Service

```bash
gcloud run deploy document-mcp \
  --source . \
  --region asia-east1 \
  --project $PROJECT_ID \
  --allow-unauthenticated \
  --platform managed \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 10 \
  --port 8080 \
  --set-env-vars "DOCUMENTS_STORAGE_PATH=/data/documents_storage,LOG_LEVEL=info,SERVER_URL=https://document-mcp-451560119112.asia-east1.run.app" \
  --set-secrets "GOOGLE_OAUTH_CLIENT_ID=GOOGLE_OAUTH_CLIENT_ID:latest,GOOGLE_OAUTH_CLIENT_SECRET=GOOGLE_OAUTH_CLIENT_SECRET:latest"
```

## Cost Breakdown

| Resource | Cost |
|----------|------|
| Cloud Run (scales to 0) | $0 when idle, ~$0.02/hour when running |
| Firestore (free tier) | $0 for <1GB storage, 50K reads/day |
| Secret Manager | ~$0.03/month for 2 secrets |
| Artifact Registry | ~$0.10/GB/month for container images |
| **Total (idle)** | **~$0.05/month** |

## Optional: Redis + VPC Connector (Higher Performance)

Only needed if you require:
- Sub-millisecond session storage
- High-frequency token refresh

**Warning: This adds ~$3.50 USD/day (~$27 HKD/day) in costs!**

```bash
# Create VPC connector (COSTS ~$0.40/day)
gcloud compute networks vpc-access connectors create document-mcp-connector \
  --region=asia-east1 \
  --network=default \
  --range=10.9.0.0/28 \
  --machine-type=e2-micro \
  --min-instances=2 \
  --max-instances=10 \
  --project=$PROJECT_ID

# Create Redis instance (COSTS ~$1.18/day)
gcloud redis instances create document-mcp-redis \
  --size=1 \
  --region=asia-east1 \
  --redis-version=redis_7_0 \
  --tier=basic \
  --network=default \
  --project=$PROJECT_ID

# Get Redis host
REDIS_HOST=$(gcloud redis instances describe document-mcp-redis \
  --region=asia-east1 --project=$PROJECT_ID --format="value(host)")

# Redeploy with VPC connector
gcloud run deploy document-mcp \
  --source . \
  --region asia-east1 \
  --project $PROJECT_ID \
  --allow-unauthenticated \
  --platform managed \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 10 \
  --port 8080 \
  --vpc-connector=document-mcp-connector \
  --set-env-vars "DOCUMENTS_STORAGE_PATH=/data/documents_storage,LOG_LEVEL=info,SERVER_URL=https://document-mcp-451560119112.asia-east1.run.app,REDIS_HOST=$REDIS_HOST" \
  --set-secrets "GOOGLE_OAUTH_CLIENT_ID=GOOGLE_OAUTH_CLIENT_ID:latest,GOOGLE_OAUTH_CLIENT_SECRET=GOOGLE_OAUTH_CLIENT_SECRET:latest"
```

## Cleanup (Delete All Resources)

To delete everything and stop all costs:

```bash
export PROJECT_ID="document-mcp-54749"

# Delete Cloud Run service
gcloud run services delete document-mcp --region=asia-east1 --project=$PROJECT_ID --quiet

# Delete VPC connectors (if created)
gcloud compute networks vpc-access connectors delete document-mcp-connector \
  --region=asia-east1 --project=$PROJECT_ID --quiet 2>/dev/null || true
gcloud compute networks vpc-access connectors delete document-mcp-connector \
  --region=us-central1 --project=$PROJECT_ID --quiet 2>/dev/null || true
gcloud compute networks vpc-access connectors delete mcp-vpc-connector \
  --region=us-central1 --project=$PROJECT_ID --quiet 2>/dev/null || true

# Delete Redis instances (if created)
gcloud redis instances delete document-mcp-redis \
  --region=asia-east1 --project=$PROJECT_ID --quiet 2>/dev/null || true
gcloud redis instances delete mcp-oauth-store \
  --region=us-central1 --project=$PROJECT_ID --quiet 2>/dev/null || true

# Clean up old container images (keep latest)
gcloud artifacts docker images list asia-east1-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/document-mcp \
  --include-tags --format="value(version)" --sort-by="~createTime" | tail -n +2 | \
  xargs -I {} gcloud artifacts docker images delete \
  "asia-east1-docker.pkg.dev/$PROJECT_ID/cloud-run-source-deploy/document-mcp@{}" \
  --project=$PROJECT_ID --quiet --delete-tags

echo "Cleanup complete. Monthly cost: ~$0"
```

## Verify Deployment

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe document-mcp \
  --region=asia-east1 --project=$PROJECT_ID --format="value(status.url)")

echo "Service URL: $SERVICE_URL"

# Health check
curl -f "${SERVICE_URL}/health"

# OAuth metadata
curl -f "${SERVICE_URL}/.well-known/oauth-authorization-server"
```

## Monitoring

```bash
# View logs
gcloud run services logs read document-mcp --region=asia-east1 --project=$PROJECT_ID --limit=50

# Check service status
gcloud run services describe document-mcp --region=asia-east1 --project=$PROJECT_ID
```
