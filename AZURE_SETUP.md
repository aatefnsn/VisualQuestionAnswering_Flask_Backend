# Azure Container Apps Deployment Guide

## Prerequisites
- Azure CLI installed
- Logged in: `az login`
- Container registry: `vqaregistry.azurecr.io`

## One-Time Infrastructure Setup

### 1. Create Storage Account and File Share
```powershell
# Create storage account
az storage account create --name vqastorage --resource-group vqa-rg --location eastus --sku Standard_LRS

# Create file share for BERT model caching
az storage share create --name bertcache --account-name vqastorage

# Get storage key (save this!)
az storage account keys list --resource-group vqa-rg --account-name vqastorage --query "[0].value" -o tsv
```

### 2. Register Storage with Container Apps Environment
```powershell
az containerapp env storage set \
  --name vqa-env \
  --resource-group vqa-rg \
  --storage-name bertcache \
  --azure-file-account-name vqastorage \
  --azure-file-account-key "PASTE_KEY_HERE" \
  --azure-file-share-name bertcache \
  --access-mode ReadWrite
```

## Deploy/Update Application

### 1. Build Image
```powershell
cd VisualQuestionAnswering_Flask_Backend
az acr build --registry vqaregistry --image vqa-backend:latest --platform linux/amd64 .
```

### 2. Update Container App
```powershell
az containerapp update \
  --name vqa-backend \
  --resource-group vqa-rg \
  --image vqaregistry.azurecr.io/vqa-backend:latest
```

### 3. Add Volume Mount (if not already configured)
```powershell
az containerapp update \
  --name vqa-backend \
  --resource-group vqa-rg \
  --yaml-patch '[{"op":"add","path":"/properties/template/volumes","value":[{"name":"bertcache","storageType":"AzureFile","storageName":"bertcache"}]},{"op":"add","path":"/properties/template/containers/0/volumeMounts","value":[{"volumeName":"bertcache","mountPath":"/mnt/bertcache"}]}]'
```

## Verify Deployment

### Test the API
```powershell
curl.exe -X POST https://vqa-backend.victoriousocean-5c59fa05.eastus.azurecontainerapps.io/predict `
  -F "file=@test/COCO_train2014_000000000081.jpg" `
  -F "question=what color is the bus?"
```

### Check Logs
```powershell
az containerapp logs show --name vqa-backend --resource-group vqa-rg --tail 50
```

## Configuration Details

### Environment Variables (set in Dockerfile)
- `PORT=8080` - Gunicorn port
- `TRANSFORMERS_CACHE=/mnt/bertcache` - BERT model cache location
- `HF_HOME=/mnt/bertcache` - Hugging Face home directory

### Volume Mount
- Azure Files share: `bertcache`
- Mount path: `/mnt/bertcache`
- Purpose: Persist BERT model across container restarts

### Resource Configuration
- CPU: 4 cores
- Memory: 8GB
- Scale: 0-20 replicas
- Estimated cost: $3-5/month with scale-to-zero

## Troubleshooting

### Container won't start
```powershell
az containerapp logs show --name vqa-backend --resource-group vqa-rg --tail 100
```

### Force new revision
```powershell
az containerapp revision restart --name vqa-backend --resource-group vqa-rg
```

### Check current image
```powershell
az containerapp show --name vqa-backend --resource-group vqa-rg --query "properties.template.containers[0].image"
```
