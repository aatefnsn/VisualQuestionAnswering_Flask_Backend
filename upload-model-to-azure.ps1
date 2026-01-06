# Upload model checkpoint to Azure Blob Storage
# This script uploads the trained model to Azure for use during Docker builds

$RESOURCE_GROUP = "vqa-rg"
$STORAGE_ACCOUNT = "vqastorage6305"
$CONTAINER_NAME = "models"
$MODEL_FILE = "app/checkpoint_17_Ahmed_768_new.pth.tar"
$BLOB_NAME = "checkpoint_17_Ahmed_768_new.pth.tar"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Uploading Model to Azure Blob Storage" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Verify model file exists
if (-not (Test-Path $MODEL_FILE)) {
    Write-Host "ERROR: Model file not found: $MODEL_FILE" -ForegroundColor Red
    exit 1
}

$fileSize = (Get-Item $MODEL_FILE).Length / 1MB
Write-Host "`nModel file: $MODEL_FILE" -ForegroundColor Green
Write-Host "Size: $fileSize MB" -ForegroundColor Green

# Get storage account key
Write-Host "`nRetrieving storage account key..." -ForegroundColor Yellow
$STORAGE_KEY = az storage account keys list `
    --resource-group $RESOURCE_GROUP `
    --account-name $STORAGE_ACCOUNT `
    --query "[0].value" -o tsv

if (-not $STORAGE_KEY) {
    Write-Host "ERROR: Failed to retrieve storage account key" -ForegroundColor Red
    exit 1
}

# Create container if it doesn't exist
Write-Host "Creating container '$CONTAINER_NAME' if needed..." -ForegroundColor Yellow
az storage container create `
    --name $CONTAINER_NAME `
    --account-name $STORAGE_ACCOUNT `
    --account-key $STORAGE_KEY `
    --output none 2>&1 | ForEach-Object { 
        if ($_ -like "*already exists*") {
            Write-Host "Container already exists" -ForegroundColor Green
        }
    }

# Upload the model
Write-Host "Uploading model to Azure..." -ForegroundColor Yellow
az storage blob upload `
    --file $MODEL_FILE `
    --container-name $CONTAINER_NAME `
    --name $BLOB_NAME `
    --account-name $STORAGE_ACCOUNT `
    --account-key $STORAGE_KEY `
    --overwrite

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nUpload successful!" -ForegroundColor Green
    Write-Host "Blob: $BLOB_NAME" -ForegroundColor Green
    Write-Host "Container: $CONTAINER_NAME" -ForegroundColor Green
    Write-Host "Storage Account: $STORAGE_ACCOUNT" -ForegroundColor Green
    
    # Get the blob URL
    $BLOB_URL = "https://$STORAGE_ACCOUNT.blob.core.windows.net/$CONTAINER_NAME/$BLOB_NAME"
    Write-Host "`nBlob URL: $BLOB_URL" -ForegroundColor Green
} else {
    Write-Host "`nERROR: Upload failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`nDone!" -ForegroundColor Cyan
