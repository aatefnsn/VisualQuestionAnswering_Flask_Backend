# Azure Event Hub Setup Checklist

## Quick Setup (5 minutes)

### 1. Create Event Hub Namespace
```
Azure Portal → Event Hubs → Create
- Name: vqa-events
- Region: East US
- Pricing Tier: Standard
```

### 2. Create Event Hub Instance
```
In vqa-events namespace → Event Hubs → New
- Name: vqa-predictions
- Partitions: 4
- Message Retention: 1 day
```

### 3. Get Connection String
```
vqa-events → Shared access policies → RootManageSharedAccessKey
Copy the "Connection string–primary key"

Example format:
Endpoint=sb://vqa-events.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=abcd1234xyz=
```

### 4. Add to GitHub Actions Secrets
```
GitHub Repo → Settings → Secrets → New repository secret

Name: EVENT_HUB_CONNECTION_STRING
Value: <paste connection string from Step 3>

Name: EVENT_HUB_NAME
Value: vqa-predictions
```

### 5. Add to Databricks Secrets
```bash
# In terminal with Databricks CLI installed:
databricks secrets create-scope --scope vqa

databricks secrets put --scope vqa --key event_hub_connection_string
# Then paste the connection string when prompted
```

---

## Verify Setup

### Test Event Hub Connection
```python
# In Python:
from azure.eventhub import EventHubProducerClient, EventData
import json

connection_str = "YOUR_CONNECTION_STRING"
producer = EventHubProducerClient.from_connection_string(connection_str, 'vqa-predictions')
test_event = EventData(json.dumps({"test": "message"}))
producer.send_batch([test_event])
producer.close()
print("✓ Event Hub is working!")
```

### Check Event Hub Metrics
```
Azure Portal → vqa-events → vqa-predictions
- Incoming Messages: Should show activity
- Outgoing Messages: Databricks should be consuming
```

---

## Cost Estimate
- **Event Hub**: $30-50/month (depends on message throughput)
- Includes: 1 consumer group, 4 partitions, 1 day retention

## FAQ

**Q: How do I know if messages are flowing?**
A: Check Azure Portal → Event Hub → Metrics tab. Should see "Incoming Messages" and "Outgoing Messages" graphs.

**Q: Can I use Event Hub free tier?**
A: Basic tier ($10/month) works but has lower throughput. Standard ($50/month) recommended for production.

**Q: What if predictions stop showing up?**
A: 
1. Verify connection string in Databricks secrets
2. Check Flask logs to see if logging is working
3. Verify Event Hub doesn't have network restrictions

**Q: How long are messages kept?**
A: Default is 1 day. For longer retention, upgrade Event Hub or export to storage.
