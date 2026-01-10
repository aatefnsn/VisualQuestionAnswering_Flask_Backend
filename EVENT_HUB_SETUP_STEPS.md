# Azure Event Hub Setup - Step by Step

## Part 1: Create Event Hub Namespace

1. Go to **Azure Portal** → https://portal.azure.com
2. Click **+ Create a resource**
3. Search for **"Event Hubs"** and click the result
4. Click **Create**
5. Fill in the form:
   - **Subscription**: Select your subscription
   - **Resource Group**: `vqa-rg` (or your existing resource group)
   - **Namespace name**: `vqa-events` (must be globally unique)
   - **Location**: Same as your other resources (e.g., `East US`)
   - **Pricing tier**: `Standard` (supports ~1MB/s throughput)
   - **Throughput units**: `1` (can increase later if needed)
6. Click **Review + Create** → **Create**
7. Wait for deployment (2-3 minutes)

## Part 2: Create Event Hub Inside Namespace

1. After namespace is created, click **Go to resource**
2. In the left sidebar, click **Event Hubs** (under Entities section)
3. Click **+ Event Hub**
4. Fill in:
   - **Name**: `vqa-predictions`
   - **Partition Count**: `1`
   - **Message Retention**: `1` (day is fine for testing)
   - **Capture**: `Off` (can enable later for archival)
5. Click **Create**
6. Wait for it to appear in the list (30 seconds)

## Part 3: Get Connection String

1. In the namespace, click **Shared access policies** (left sidebar)
2. Look for policy named `RootManageSharedAccessKey` (or create new one if needed)
3. If creating new policy:
   - Click **+ Add**
   - Name: `VQABackendPolicy`
   - Check: **Listen**, **Send**, **Manage**
   - Click **Create**
4. Click on the policy (RootManageSharedAccessKey or VQABackendPolicy)
5. Copy the **Primary Connection String** (it will look like):
   ```
   Endpoint=sb://vqa-events.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=xxxxx...
   ```
6. **Save this value** - you'll need it for GitHub

## Part 4: Get Event Hub Name

The **Event Hub name** is simply:
```
vqa-predictions
```

## Part 5: Add Secrets to GitHub

1. Go to your GitHub repository
2. Click **Settings** (top right)
3. Click **Secrets and variables** → **Actions** (left sidebar)
4. Click **New repository secret**

### First Secret:

5. **Name**: `EVENT_HUB_CONNECTION_STRING`
6. **Value**: Paste the entire connection string from Part 3
7. Click **Add secret**

### Second Secret:

8. Click **New repository secret** again
9. **Name**: `EVENT_HUB_NAME`
10. **Value**: `vqa-predictions`
11. Click **Add secret**

## Part 6: Verify Secrets in GitHub

1. Back on Secrets page, you should see:
   - ✅ `EVENT_HUB_CONNECTION_STRING` (value hidden)
   - ✅ `EVENT_HUB_NAME` (value hidden)

## Part 7: Ready to Deploy!

Now you can trigger the GitHub Actions workflow:

```bash
git add .
git commit -m "Configure Event Hub secrets for deployment"
git push origin main
```

The workflow will:
1. Build Docker image (secrets NOT stored in image)
2. Deploy to Container Apps
3. Inject `EVENT_HUB_CONNECTION_STRING` and `EVENT_HUB_NAME` as environment variables
4. Flask app reads them with `os.getenv()` at startup
5. Predictions will be logged to Event Hub automatically

## Troubleshooting

### Connection string not working?
- Verify it starts with `Endpoint=sb://`
- Check that the Event Hub namespace name matches (`vqa-events`)
- Make sure you copied the **Primary** Connection String (not secondary)

### Event Hub name wrong?
- Must match exactly: `vqa-predictions` (case-sensitive in some cases)
- Check it's created in the correct namespace

### Secrets not passed to container?
- Verify secrets are in GitHub repo settings (not organization level)
- Wait 10 seconds after adding secrets before triggering workflow
- Check workflow logs for secret verification step

### Permission errors?
- Make sure shared access policy has **Send** permission
- Check your Azure subscription has permissions to create Event Hubs

## Quick Reference

| Item | Value |
|------|-------|
| Namespace | `vqa-events` |
| Event Hub | `vqa-predictions` |
| GitHub Secret 1 | `EVENT_HUB_CONNECTION_STRING` |
| GitHub Secret 2 | `EVENT_HUB_NAME` |
| Connection String Format | `Endpoint=sb://vqa-events.servicebus.windows.net/;SharedAccessKeyName=...;SharedAccessKey=...` |
