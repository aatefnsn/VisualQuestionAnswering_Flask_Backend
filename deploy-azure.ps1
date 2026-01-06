# Azure VQA Backend Deployment Script
# Run this script from the VisualQuestionAnswering_Flask_Backend directory

# Stop on any error
$ErrorActionPreference = "Stop"

$RG = "vqa-rg"
$ENV = "vqa-env"
$APP = "vqa-backend"
$STORAGE = "vqastorage$(Get-Random -Minimum 1000 -Maximum 9999)"  # Make storage name unique
$REGISTRY = "vqaregistry"
$LOCATION = "eastus"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Azure VQA Backend Deployment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# ============================================
# PART 0: VERIFY SUBSCRIPTION
# ============================================

Write-Host "`n[0/9] Checking Azure subscription..." -ForegroundColor Yellow
$currentSub = az account show --query "name" -o tsv 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Not logged into Azure! Run: az login" -ForegroundColor Red
    exit 1
}
Write-Host "Current subscription: $currentSub" -ForegroundColor Green

# Verify resource group exists
Write-Host "Verifying resource group exists..." -ForegroundColor Yellow
$rgExists = az group show --name $RG --query "name" -o tsv 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Resource group '$RG' not found!" -ForegroundColor Red
    Write-Host "Available subscriptions:" -ForegroundColor Yellow
    az account list --query "[].{Name:name, ID:id, State:state}" -o table
    Write-Host "`nTo switch subscription: az account set --subscription <subscription-id>" -ForegroundColor Yellow
    exit 1
}
Write-Host "Resource group verified: $RG" -ForegroundColor Green

# Register required Azure providers
Write-Host "`nRegistering required Azure resource providers..." -ForegroundColor Yellow
$providers = @(
    "Microsoft.Storage",
    "Microsoft.ContainerRegistry",
    "Microsoft.App",
    "Microsoft.OperationalInsights"
)

foreach ($provider in $providers) {
    $state = az provider show --namespace $provider --query "registrationState" -o tsv 2>$null
    if ($state -eq "Registered") {
        Write-Host "  [OK] $provider already registered" -ForegroundColor Green
    } elseif ($state -eq "Registering") {
        Write-Host "  [WAIT] $provider is registering..." -ForegroundColor Yellow
    } else {
        Write-Host "  [->] Registering $provider..." -ForegroundColor Yellow
        az provider register --namespace $provider --wait
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  [OK] $provider registered successfully" -ForegroundColor Green
        } else {
            Write-Host "  [ERROR] Failed to register $provider" -ForegroundColor Red
        }
    }
}
Write-Host "All providers ready!" -ForegroundColor Green

# ============================================
# PART 1: ONE-TIME INFRASTRUCTURE SETUP
# (Skip if already exists - SKIPPED FOR THIS RUN)
# ============================================

Write-Host "`n[1-4/8] Skipping storage setup (already configured)..." -ForegroundColor Yellow

# ============================================
# PART 2: BUILD AND DEPLOY APPLICATION
# (Run every time you update code)
# ============================================

Write-Host "`n[5/8] Building Docker image in Azure Container Registry..." -ForegroundColor Yellow
az acr build --registry $REGISTRY --image vqa-backend:latest --platform linux/amd64 .
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to build Docker image!" -ForegroundColor Red
    exit 1
}

Write-Host "[6/8] Updating Container App with new image..." -ForegroundColor Yellow
az containerapp update `
  --name $APP `
  --resource-group $RG `
  --image "$REGISTRY.azurecr.io/vqa-backend:latest"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to update container app!" -ForegroundColor Red
    exit 1
}

Write-Host "[7/8] Configuring volume mount for BERT cache..." -ForegroundColor Yellow
# Check if volume is already configured
$currentConfig = az containerapp show --name $APP --resource-group $RG --query "properties.template.volumes" -o json 2>$null
if ($currentConfig -and $currentConfig -like "*bertcache*") {
    Write-Host "Volume mount already configured, skipping." -ForegroundColor Green
} else {
    Write-Host "WARNING: Automatic volume mount configuration not supported in current Azure CLI version." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please configure volume mount manually via Azure Portal:" -ForegroundColor Cyan
    Write-Host "1. Go to: https://portal.azure.com" -ForegroundColor White
    Write-Host "2. Navigate to: Container Apps > vqa-backend > Volumes" -ForegroundColor White
    Write-Host "3. Click 'Add' and configure:" -ForegroundColor White
    Write-Host "   - Volume name: bertcache" -ForegroundColor White
    Write-Host "   - Storage type: Azure Files" -ForegroundColor White
    Write-Host "   - Storage account: $STORAGE" -ForegroundColor White
    Write-Host "   - File share: bertcache" -ForegroundColor White
    Write-Host "   - Mount path: /mnt/bertcache" -ForegroundColor White
    Write-Host ""
    Write-Host "BENEFIT: BERT model will persist across container restarts (no re-download)" -ForegroundColor Green
    Write-Host "SKIP: If you're okay with BERT downloading on each cold start (~10 seconds)" -ForegroundColor Yellow
}

Write-Host "[8/8] Deployment complete!" -ForegroundColor Green

# ============================================
# DISPLAY INFORMATION
# ============================================

$URL = az containerapp show --name $APP --resource-group $RG --query "properties.configuration.ingress.fqdn" -o tsv

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Deployment Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "App Name: $APP" -ForegroundColor White
Write-Host "URL: https://$URL" -ForegroundColor White
Write-Host "Storage: $STORAGE (bertcache)" -ForegroundColor White
Write-Host "`nTest command:" -ForegroundColor Yellow
Write-Host "curl.exe -X POST https://$URL/predict -F `"file=@test/COCO_train2014_000000000081.jpg`" -F `"question=what color is the bus?`"" -ForegroundColor Gray
Write-Host "`nView logs:" -ForegroundColor Yellow
Write-Host "az containerapp logs show --name $APP --resource-group $RG --tail 50" -ForegroundColor Gray
Write-Host "========================================`n" -ForegroundColor Cyan
