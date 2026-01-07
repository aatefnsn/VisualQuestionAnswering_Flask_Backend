# VQA Backend Deployment Pipeline

**Last Updated:** January 6, 2026

## Overview

This deployment pipeline includes comprehensive testing, health checks, and automatic rollback capabilities to ensure reliable deployments to Azure Container Apps.

## Deployment Workflow

### 1. **Unit Testing** (Local + GitHub Actions)
   - Tests Flask app initialization
   - Validates API endpoints
   - Checks file upload handling
   - Verifies CORS headers
   - Ensures error handling works correctly

### 2. **Docker Build** (Azure Container Registry)
   - Builds optimized Docker image
   - Downloads model checkpoint from Azure Blob Storage
   - Packages all dependencies
   - Creates tagged image with semantic versioning (v1, v2, v3, etc.)

### 3. **Deployment** (Azure Container Apps)
   - Deploys new container version
   - Tracks previous version for rollback
   - Configures volumes and environment variables

### 4. **Health Checks** (Post-Deployment)
   - Tests `/health` endpoint (up to 120 seconds)
   - Tests `/predict` endpoint for responsiveness
   - Validates container is properly responding

### 5. **Automatic Rollback** (If Tests Fail)
   - Automatically reverts to previous stable version
   - Sends notification with failure reason
   - Preserves service availability

## Running Locally

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Unit Tests
```bash
python -m unittest test_backend.py -v
# or with pytest
pytest test_backend.py -v
```

### Test the API Locally
```bash
python main.py
# Then test endpoints:
curl http://localhost:5000/health
curl -X POST http://localhost:5000/predict -F "question=test"
```

## GitHub Actions Workflow

### File Location
`.github/workflows/deploy.yml`

### Workflow Jobs

#### 1. **test** job
- Runs on: `ubuntu-latest`
- Steps:
  - Checkout code
  - Set up Python 3.10
  - Install dependencies
  - Run unit tests
  - Report results

#### 2. **build-and-deploy** job
- Runs on: `ubuntu-latest`
- Depends on: `test` job (waits for tests to pass)
- Steps:
  - Checkout code
  - Azure login
  - Version management (semantic versioning)
  - Validate Dockerfile
  - Build image in ACR
  - Deploy to Container Apps
  - Wait for startup (15 seconds)
  - Health check (max 120 seconds, retries every 10 seconds)
  - Endpoint testing
  - Rollback if health checks fail
  - Deployment summary

### Triggering Deployment

#### Automatic (on push to main)
```bash
git push origin main
```

#### Manual (from GitHub UI)
1. Go to Actions tab
2. Select "Deploy to Azure Container Apps"
3. Click "Run workflow"

## Azure Resources

- **Resource Group:** `vqa-rg`
- **Container Registry:** `vqaregistry`
- **Container App:** `vqa-backend`
- **Storage Account:** `vqastorage6305`
- **Model Storage:** `vqastorage6305/models/checkpoint_17_Ahmed_768_new.pth.tar`

## Endpoints

### Health Check
```bash
GET /health
```
Response:
```json
{
  "status": "healthy",
  "service": "vqa-backend"
}
```

### Prediction
```bash
POST /predict
Content-Type: multipart/form-data

Parameters:
- file: image file (jpg, jpeg, png)
- question: question text
```

## Model Management

### Upload New Model
```bash
./upload-model-to-azure.ps1
```

This script:
1. Checks for model checkpoint
2. Connects to Azure Storage
3. Creates "models" container if needed
4. Uploads checkpoint to Azure Blob Storage
5. Displays blob URL for reference

The Dockerfile automatically downloads this model during container build.

## Versioning

Deployments use semantic versioning:
- `v1`, `v2`, `v3`, etc.
- Each deployment increments the version
- Previous version is kept for rollback

### View Image Tags
```bash
az acr repository show-tags --name vqaregistry --repository vqa-backend --orderby time_desc
```

## Rollback Procedure

### Automatic Rollback
- Triggered automatically if health checks fail
- Previous image is deployed within minutes
- GitHub Actions notifies via step summary

### Manual Rollback
```bash
# Get previous image tag
az acr repository show-tags --name vqaregistry --repository vqa-backend

# Rollback to previous version
az containerapp update \
  --name vqa-backend \
  --resource-group vqa-rg \
  --image vqaregistry.azurecr.io/vqa-backend:v1
```

## Monitoring

### View Container Logs
```bash
az containerapp logs show \
  --name vqa-backend \
  --resource-group vqa-rg \
  --tail 50
```

### View GitHub Actions
- Go to repository Actions tab
- Click on workflow run
- Review step-by-step logs

## Troubleshooting

### Model Download Fails
- Check Azure Blob Storage connection
- Verify model file exists: `https://vqastorage6305.blob.core.windows.net/models/checkpoint_17_Ahmed_768_new.pth.tar`
- Check storage account key permissions

### Health Check Fails
- Container may still be starting (takes 15-30 seconds)
- Check container logs: `az containerapp logs show --name vqa-backend --resource-group vqa-rg`
- Verify Flask app starts without errors

### Rollback Didn't Work
- No previous version exists (first deployment)
- Previous image not in ACR registry
- Manual rollback needed

## Next Steps

1. **Test in staging** - Create separate staging environment before production
2. **Add metrics** - Monitor response times, error rates, resource usage
3. **Scale configuration** - Adjust min/max replicas based on load
4. **Database integration** - Add logging/analytics backend
5. **Custom domain** - Set up custom domain with SSL certificate
