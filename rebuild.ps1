# Quick Rebuild Script - Use this for code updates
# Prerequisites: Infrastructure already set up via deploy-azure.ps1

$RG = "vqa-rg"
$APP = "vqa-backend"
$REGISTRY = "vqaregistry"

Write-Host "Rebuilding VQA Backend..." -ForegroundColor Cyan

Write-Host "[1/2] Building image in ACR..." -ForegroundColor Yellow
az acr build --registry $REGISTRY --image vqa-backend:latest --platform linux/amd64 .

Write-Host "[2/2] Updating container app..." -ForegroundColor Yellow
az containerapp update --name $APP --resource-group $RG --image "$REGISTRY.azurecr.io/vqa-backend:latest"

$URL = az containerapp show --name $APP --resource-group $RG --query "properties.configuration.ingress.fqdn" -o tsv
Write-Host "`nDeployment complete! URL: https://$URL" -ForegroundColor Green
