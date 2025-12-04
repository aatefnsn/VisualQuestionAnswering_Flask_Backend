# Quick Rebuild Script - Use this for code updates
# Prerequisites: Infrastructure already set up via deploy-azure.ps1

$RG = "vqa-rg"
$APP = "vqa-backend"
$REGISTRY = "vqaregistry"

Write-Host "Rebuilding VQA Backend..." -ForegroundColor Cyan

# Get all tags from ACR and find the latest version number
Write-Host "Fetching latest version from ACR..." -ForegroundColor Yellow
$tags = az acr repository show-tags --name $REGISTRY --repository vqa-backend --orderby time_desc --output json 2>$null | ConvertFrom-Json

$latestVersion = 0
if ($tags) {
    foreach ($tag in $tags) {
        if ($tag -match '^v(\d+)$') {
            $versionNum = [int]$matches[1]
            if ($versionNum -gt $latestVersion) {
                $latestVersion = $versionNum
            }
        }
    }
}

$newVersion = $latestVersion + 1
$Version = "v$newVersion"

Write-Host "Current version: v$latestVersion" -ForegroundColor Cyan
Write-Host "New version: $Version" -ForegroundColor Green

Write-Host "[1/2] Building image in ACR..." -ForegroundColor Yellow
az acr build --registry $REGISTRY --image "vqa-backend:$Version" --platform linux/amd64 .

Write-Host "[2/2] Updating container app..." -ForegroundColor Yellow
az containerapp update --name $APP --resource-group $RG --image "$REGISTRY.azurecr.io/vqa-backend:$Version"

$URL = az containerapp show --name $APP --resource-group $RG --query "properties.configuration.ingress.fqdn" -o tsv
Write-Host "`nDeployment complete! URL: https://$URL" -ForegroundColor Green
Write-Host "Deployed image: $REGISTRY.azurecr.io/vqa-backend:$Version" -ForegroundColor Cyan
