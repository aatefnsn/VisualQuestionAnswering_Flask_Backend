# Azure Container Apps Migration Guide

## Overview
This document details the migration of the Visual Question Answering (VQA) Flask backend from Google Cloud Run to Azure Container Apps, completed in December 2025.

## Migration Summary

### Original Setup (GCP)
- **Platform:** Google Cloud Run
- **Configuration:** 2 CPU, 8GB RAM
- **Cost:** ~$5/month with auto-scaling
- **Deployment:** Direct container deployment with gcloud CLI

### New Setup (Azure)
- **Platform:** Azure Container Apps
- **Configuration:** 4 CPU, 8GB RAM (Azure's minimum for 8GB)
- **Cost Target:** <$5/month with scale-to-zero (0-20 replicas)
- **URL:** https://vqa-backend.victoriousocean-5c59fa05.eastus.azurecontainerapps.io

## Key Changes Made

### 1. PyTorch Compatibility Fix
**Problem:** PyTorch 1.12.0 had executable stack security issues on Azure Container Apps
```
Error: libtorch_cpu.so: cannot enable executable stack as shared object requires: Invalid argument
```

**Solution:** Upgraded to PyTorch 2.0.1
- **File:** `requirements.txt`
- **Changes:**
  ```diff
  - torch==1.12.0+cpu
  - torchvision==0.13.0+cpu
  - --extra-index-url https://download.pytorch.org/whl/cpu
  + torch==2.0.1
  + torchvision==0.15.2
  ```

### 2. PORT Environment Variable
**Problem:** Gunicorn couldn't start because $PORT was undefined

**Solution:** Added PORT environment variable to Dockerfile
- **File:** `Dockerfile`
- **Change:**
  ```dockerfile
  ENV PORT 8080
  ```

### 3. Model Caching with Azure Files
**Problem:** Models (BERT 440MB, ResNet18 44.7MB, VQA checkpoint 526MB) re-downloaded on every cold start, causing 55+ second startup times

**Solution:** Implemented persistent storage with Azure Files volume mount
- **Azure Resources Created:**
  - Storage Account: `vqastorage####` (random suffix)
  - File Share: `bertcache`
  - Mounted at: `/mnt/bertcache`

- **File:** `Dockerfile`
- **Changes:**
  ```dockerfile
  ENV TRANSFORMERS_CACHE /mnt/bertcache
  ENV HF_HOME /mnt/bertcache
  ENV TORCH_HOME /mnt/bertcache
  ```

### 4. Deployment Automation
**New Files Created:**

#### `deploy-azure.ps1`
Full deployment script that handles:
- Azure provider registration (Microsoft.Storage, Microsoft.ContainerRegistry, Microsoft.App, Microsoft.OperationalInsights)
- Resource group creation (vqa-rg)
- Container Apps environment setup (vqa-env)
- Azure Container Registry setup (vqaregistry)
- Storage account and file share creation
- Storage registration with Container Apps environment
- Image building in ACR with linux/amd64 platform
- Container app deployment with configuration
- Volume mount setup instructions (manual via Portal due to CLI limitations)

#### `rebuild.ps1`
Quick rebuild script for code updates:
- Rebuilds image in ACR
- Updates container app with new image
- No infrastructure changes

#### `AZURE_SETUP.md`
Manual deployment documentation with:
- Step-by-step setup instructions
- Volume mount configuration via Azure Portal
- Testing and verification steps
- Troubleshooting guide

### 5. Code Cleanup
**Removed Legacy Files:**
- `app/main.py` (duplicate, renamed to `_old`)
- `app/torch_utils.py` (duplicate, renamed to `_old`)
- `wsgi.py` (Heroku/GCP-specific, not used in Azure)
- `Procfile` (Heroku/GCP-specific, not used in Azure)

**Reason:** Azure Container Apps uses the Dockerfile CMD directly:
```dockerfile
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
```

## Architecture Comparison

### GCP Cloud Run
```
User Request → Cloud Run → Container (transient) → Response
                ↓
          Downloads models on every cold start
```

### Azure Container Apps
```
User Request → Container Apps → Container → Response
                                    ↓
                              Azure Files Mount
                              /mnt/bertcache
                              (persistent storage)
                                    ↓
                         Models cached across restarts
```

## Performance Metrics

### Cold Start Times
- **Before caching:** 55+ seconds
- **After caching (expected):** 30-40 seconds
  - Container start: ~15s
  - Model loading from cache: ~10s
  - Initialization: ~10s
  - First inference: ~10s

### Model Sizes
- BERT base-uncased: 440MB
- ResNet18: 44.7MB
- VQA checkpoint: 526MB
- **Total cached:** ~1GB

## Deployment Commands

### Full Deployment
```powershell
powershell -ExecutionPolicy Bypass -File .\deploy-azure.ps1
```

### Quick Rebuild (code changes only)
```powershell
powershell -ExecutionPolicy Bypass -File .\rebuild.ps1
```

### Test Endpoint
```powershell
curl -Method POST `
  -Uri "https://vqa-backend.victoriousocean-5c59fa05.eastus.azurecontainerapps.io/predict" `
  -Form @{
    file = Get-Item "path\to\image.jpg"
    question = "what color is the shirt?"
  }
```

### View Logs
```powershell
az containerapp logs show `
  --name vqa-backend `
  --resource-group vqa-rg `
  --tail 100 `
  --follow
```

### Check Volume Mount Contents
```powershell
az containerapp exec `
  --name vqa-backend `
  --resource-group vqa-rg `
  --command "ls -lah /mnt/bertcache"
```

## Azure Resources Created

| Resource Type | Name | Purpose |
|---------------|------|---------|
| Resource Group | vqa-rg | Container for all resources |
| Container Apps Environment | vqa-env | Managed environment for container apps |
| Container Registry | vqaregistry | Stores container images |
| Storage Account | vqastorage#### | Provides file share storage |
| File Share | bertcache | Persistent model cache |
| Container App | vqa-backend | Runs the Flask application |
| Log Analytics Workspace | (auto-created) | Logging and monitoring |

## Configuration Details

### Container App Settings
- **Min Replicas:** 0 (scale-to-zero for cost savings)
- **Max Replicas:** 20
- **CPU:** 4 cores
- **Memory:** 8GB
- **Ingress:** External, port 8080
- **Authentication:** None (public API)

### Environment Variables
```dockerfile
PORT=8080
TRANSFORMERS_CACHE=/mnt/bertcache
HF_HOME=/mnt/bertcache
TORCH_HOME=/mnt/bertcache
```

### Volume Mount
- **Volume Name:** bertcache
- **Storage Type:** AzureFile
- **Mount Path:** /mnt/bertcache
- **Storage Account:** vqastorage####
- **File Share:** bertcache

## Cost Optimization

### Scale-to-Zero Strategy
- Container scales to 0 replicas after idle period
- First request triggers cold start (30-40s with caching)
- Subsequent requests are fast (1-2s)
- No charges when scaled to zero

### Storage Costs
- Azure Files Standard: ~$0.06/GB/month
- 1GB cached models: ~$0.06/month
- **Total estimated cost:** <$5/month

## Troubleshooting

### Check if models are cached
```powershell
az containerapp logs show --name vqa-backend --resource-group vqa-rg --tail 100 | Select-String -Pattern "Downloading"
```
- **Expected:** No "Downloading" messages after first run
- **Issue:** If you see downloads, check TRANSFORMERS_CACHE and TORCH_HOME environment variables

### Verify volume mount
```powershell
az containerapp show --name vqa-backend --resource-group vqa-rg --query "{volumes:properties.template.volumes,volumeMounts:properties.template.containers[0].volumeMounts}"
```
- **Expected:** volumeMounts shows `/mnt/bertcache`, volumes shows `bertcache` with `AzureFile` storage type

### Test cold start performance
```powershell
# Make request, wait for scale to zero (~15 minutes), then test again
Measure-Command {
  curl -Method POST `
    -Uri "https://vqa-backend.victoriousocean-5c59fa05.eastus.azurecontainerapps.io/predict" `
    -Form @{file = Get-Item "test.jpg"; question = "test?"}
}
```

## Known Issues & Resolutions

### Issue: Platform Architecture Mismatch
- **Problem:** Local Docker builds ARM64 images on Apple Silicon
- **Solution:** Use `az acr build` with `--platform linux/amd64` flag

### Issue: Azure CLI Volume Mount Limitations
- **Problem:** `--yaml-patch` and `--add-volume` not available in current CLI version
- **Solution:** Configure volume mounts via Azure Portal (Containers → Edit and deploy → Volume mounts)

### Issue: BERT Model Pre-download Fails in Build
- **Problem:** AMD64 build environment can't run Python pre-download script
- **Solution:** Commented out BERT pre-download in Dockerfile, rely on runtime download to persistent storage

## Migration Checklist

- [x] Update requirements.txt for PyTorch 2.0.1
- [x] Add PORT environment variable to Dockerfile
- [x] Create Azure resource group and environment
- [x] Set up Azure Container Registry
- [x] Create storage account and file share
- [x] Register storage with Container Apps environment
- [x] Add caching environment variables to Dockerfile
- [x] Configure volume mount in Container App
- [x] Create deployment scripts (deploy-azure.ps1, rebuild.ps1)
- [x] Remove legacy GCP files (wsgi.py, Procfile, app/ duplicates)
- [x] Test deployment and verify functionality
- [x] Verify model caching works
- [x] Document migration process

## Next Steps

1. **Monitor performance:** Track cold start times and costs over first month
2. **Optimize further:** Consider pre-warming strategies if cold starts are still too slow
3. **Set up alerts:** Configure Azure Monitor alerts for failures or high costs
4. **GitHub Actions:** Consider automating deployments with CI/CD pipeline
5. **Scaling policies:** Fine-tune min/max replicas based on actual usage patterns

## References

- [Azure Container Apps Documentation](https://learn.microsoft.com/en-us/azure/container-apps/)
- [Azure Files Documentation](https://learn.microsoft.com/en-us/azure/storage/files/)
- [PyTorch 2.0 Release Notes](https://pytorch.org/blog/pytorch-2.0-release/)
- Original GCP Deployment: December 2022

---

**Migration completed:** December 2025  
**Migrated by:** Ahmed  
**Original deployment:** December 2022 (Google Cloud Run)
