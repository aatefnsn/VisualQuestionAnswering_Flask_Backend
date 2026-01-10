# GitHub Actions → Event Hub Setup Checklist

## What Was Updated

The GitHub Actions workflow now automatically passes Event Hub secrets from GitHub to the Container Apps deployment.

### Changes Made to `.github/workflows/deploy.yml`:

1. **Secret Verification** (line 70-75)
   - Added checks for `EVENT_HUB_CONNECTION_STRING`
   - Added checks for `EVENT_HUB_NAME`

2. **Deploy Step** (line 164-172)
   - Now passes environment variables to container using `--set-env-vars`
   - Event Hub connection string injected at deployment time
   - Event Hub name injected at deployment time

---

## How It Works

```
GitHub Actions Workflow
    ↓
Reads secrets from GitHub
    ↓
Passes to Container Apps via az containerapp update
    ↓
Sets environment variables in running container
    ↓
Flask app reads from os.getenv()
    ↓
Event Hub logging enabled
```

---

## Required GitHub Actions Secrets

Add these to your GitHub repository:

**Settings → Secrets → New repository secret**

| Secret Name | Example Value | Where to Get |
|------------|------------------|--------------|
| `EVENT_HUB_CONNECTION_STRING` | `Endpoint=sb://vqa-events.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=xxxxx` | Azure Portal → Event Hub Namespace → Shared Access Policies |
| `EVENT_HUB_NAME` | `vqa-predictions` | Azure Portal → Event Hub Namespace → Event Hubs |

### How to Get EVENT_HUB_CONNECTION_STRING:

1. Azure Portal → Event Hubs → `vqa-events` (namespace)
2. Left menu → **Shared access policies**
3. Click **RootManageSharedAccessKey**
4. Copy **Connection string–primary key**
5. Paste into GitHub secret

### How to Get EVENT_HUB_NAME:

1. Azure Portal → Event Hubs → `vqa-events` (namespace)
2. Left menu → **Event Hubs**
3. Name of your hub (usually `vqa-predictions`)
4. Copy and paste into GitHub secret

---

## Deployment Flow with Secrets

When you push to `main` branch:

```
1. GitHub Action starts
   ↓
2. Checks all secrets exist (including Event Hub)
   ✓ EVENT_HUB_CONNECTION_STRING verified
   ✓ EVENT_HUB_NAME verified
   ↓
3. Builds Docker image in ACR
   ↓
4. Deploys to Container Apps
   ↓
5. Passes secrets as environment variables:
   - EVENT_HUB_CONNECTION_STRING=<secret value>
   - EVENT_HUB_NAME=vqa-predictions
   ↓
6. Container starts with secrets loaded
   ↓
7. Flask app reads from os.getenv()
   ↓
8. Event Hub logging enabled! ✓
```

---

## Testing the Setup

### Test 1: Verify Secrets in Workflow Log

1. Push changes to main branch
2. Go to GitHub → Actions
3. Watch the build run
4. Look for this line in the "Verify secrets are configured" step:
   ```
   ✓ All secrets are configured (including Event Hub)
   ```

### Test 2: Verify Container Has Secrets

After deployment completes:

```bash
# SSH into Container Apps or check logs:
az containerapp logs show \
  --name vqa-backend \
  --resource-group vqa-rg \
  --tail 50
```

Look for this message when predictions are made:
```
✓ Logged to Event Hub: color → yellow (56.77%)
```

### Test 3: Send Test Prediction

```bash
curl -X POST https://vqa-backend.victoriousocean-5c59fa05.eastus.azurecontainerapps.io/predict \
  -F "file=@test_image.jpg" \
  -F "question=what color is the bus?"
```

Check Azure Portal → Event Hub → Metrics:
- Should see incoming messages graph spike
- Should see outgoing messages (Databricks consuming)

---

## Troubleshooting

### Issue: "EVENT_HUB_CONNECTION_STRING not set"
**Solution:**
1. Go to GitHub → Settings → Secrets
2. Verify `EVENT_HUB_CONNECTION_STRING` exists
3. Verify value is not empty
4. Re-run workflow

### Issue: Event Hub logging fails but predictions work
**Solution:**
1. This is OK! Code is designed to fail gracefully
2. Check logs: `⚠️ Event Hub logging failed (non-critical)`
3. Verify connection string format
4. Verify Event Hub namespace name is correct

### Issue: Container starts but no Event Hub messages
**Solution:**
1. Verify secrets were passed during deployment:
   ```bash
   az containerapp show \
     --name vqa-backend \
     --resource-group vqa-rg \
     --query properties.template.containers[0].env \
     -o json
   ```
   Should show `EVENT_HUB_CONNECTION_STRING` and `EVENT_HUB_NAME`

2. Check Event Hub in Azure Portal:
   - Namespace → Event Hub → **Monitoring** tab
   - Should see incoming messages

3. Send a test prediction and watch the graph in real-time

---

## Security Notes

✅ Secrets are:
- Encrypted in GitHub
- Only visible to workflow during execution
- Never logged or printed
- Not stored in container image

❌ Secrets are NOT:
- In .env files
- In Docker images
- In logs
- In version control

---

## Next Steps

1. ✅ Add secrets to GitHub (EVENT_HUB_CONNECTION_STRING, EVENT_HUB_NAME)
2. ✅ Push latest code to main (includes workflow updates)
3. ✅ Watch GitHub Actions build
4. ✅ Verify "All secrets configured" passes
5. ✅ Make a test prediction
6. ✅ Check Event Hub metrics for incoming messages
7. ✅ Set up Databricks notebooks to consume from Event Hub

---

## Files Updated

- `.github/workflows/deploy.yml` - Deploy step now passes Event Hub secrets

---

## Quick Reference: The Workflow Change

**Before:**
```yaml
az containerapp update \
  --name vqa-backend \
  --image image:tag
```

**After:**
```yaml
az containerapp update \
  --name vqa-backend \
  --image image:tag \
  --set-env-vars \
    EVENT_HUB_CONNECTION_STRING="${{ secrets.EVENT_HUB_CONNECTION_STRING }}" \
    EVENT_HUB_NAME="${{ secrets.EVENT_HUB_NAME }}"
```

The container now has access to the secrets through environment variables!
