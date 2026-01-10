# VQA Real-Time KPI Dashboard - Implementation Guide

## Overview
This setup creates a real-time dashboard that automatically updates 5 key metrics with every new prediction.

## The 5 KPIs Tracked
1. **Total Prediction Count** - Total number of predictions made
2. **Max Confidence Score** - Highest probability prediction ever
3. **Lowest Confidence Score** - Lowest probability prediction
4. **Average Confidence** - Average of all prediction confidences
5. **Most Frequent Question Type** - Which question type appears most

---

## Architecture

```
Flask Backend (main.py)
    ↓
Categorize question + Log to Event Hub
    ↓
Azure Event Hub (vqa-predictions)
    ↓
Databricks Notebook 1 (ETL Pipeline)
    ├─ Reads from Event Hub
    └─ Writes to Delta Lake
    ↓
Databricks Notebook 2 (KPI Dashboard)
    ├─ Reads from Event Hub
    ├─ Calculates 5 KPIs
    ├─ Writes to Delta Lake (KPI metrics)
    └─ Displays live dashboard
    ↓
Live KPI Dashboard (updates every 10 seconds)
```

---

## Step-by-Step Implementation

### Phase 1: Azure Setup (30 minutes)

#### 1.1 Create Azure Event Hub
1. Azure Portal → Create "Event Hubs Namespace"
   - Name: `vqa-events`
   - Region: East US (same as other resources)
   - Pricing tier: Standard
2. Inside namespace, create Event Hub:
   - Name: `vqa-predictions`
   - Partitions: 4
3. Get connection string:
   - Namespace → Shared access policies → RootManageSharedAccessKey
   - Copy "Connection string–primary key"
   - Format: `Endpoint=sb://vqa-events.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=xxxxx`

#### 1.2 Create Databricks Secrets
In Databricks workspace:
```bash
# Create secret scope
databricks secrets create-scope --scope vqa

# Add Event Hub connection string
databricks secrets put --scope vqa --key event_hub_connection_string
# Paste the connection string from Step 1.1
```

### Phase 2: Backend Setup (15 minutes)

#### 2.1 Update Requirements
Add to `requirements.txt`:
```
azure-eventhub==5.11.5
```

#### 2.2 Add GitHub Actions Secrets
In your GitHub repository:
```
EVENT_HUB_CONNECTION_STRING: <paste from Step 1.1>
EVENT_HUB_NAME: vqa-predictions
```

#### 2.3 Deploy Updated Backend
```bash
git add main.py requirements.txt
git commit -m "Add Event Hub logging and question categorization"
git push origin main
```

### Phase 3: Databricks Setup (45 minutes)

#### 3.1 Create ETL Notebook
1. Go to Databricks workspace
2. Create new notebook: `vqa_etl_pipeline`
3. Copy content from: `databricks_etl_pipeline.py`
4. Attach to cluster (create new cluster if needed)
5. Run the notebook - it will start streaming predictions

#### 3.2 Create KPI Dashboard Notebook
1. Create new notebook: `vqa_kpi_dashboard`
2. Copy content from: `databricks_kpi_dashboard.py`
3. Attach to same cluster
4. Run the notebook - it will show live KPIs

#### 3.3 Set Up Jobs to Run Continuously
1. In Databricks: Jobs → Create job
2. Configure:
   - Job name: `VQA-ETL-Pipeline`
   - Type: Notebook
   - Notebook: `vqa_etl_pipeline`
   - Cluster: Your cluster
   - Schedule: "Never" (runs continuously)
3. Repeat for `vqa_kpi_dashboard`

---

## Testing the Setup

### Test 1: Verify Event Hub Connection
```bash
# In Flask container, test Event Hub:
python -c "
from azure.eventhub import EventHubProducerClient, EventData
import json
import os

conn_str = os.getenv('EVENT_HUB_CONNECTION_STRING')
producer = EventHubProducerClient.from_connection_string(conn_str, 'vqa-predictions')
test_event = EventData(json.dumps({'test': 'message'}))
producer.send_batch([test_event])
producer.close()
print('✓ Event Hub connection works!')
"
```

### Test 2: Send a Test Prediction
```bash
# In your local environment:
curl -X POST http://localhost:8080/predict \
  -F "file=@test_image.jpg" \
  -F "question=what color is the bus?"
```

### Test 3: Check Databricks Dashboard
1. Go to Databricks workspace
2. Open `vqa_kpi_dashboard` notebook
3. Run all cells
4. Should see:
   ```
   ============================================================
   VQA REAL-TIME KPI DASHBOARD
   ============================================================
   ✓ Last Updated: 2026-01-09 14:32:15
   
   📊 KEY METRICS:
     1️⃣  Total Predictions:        5
     2️⃣  Max Confidence Score:    78.90% (yellow)
     3️⃣  Lowest Confidence Score: 32.10% (red)
     4️⃣  Average Confidence:      62.35%
     5️⃣  Most Frequent Q Type:    color
   ============================================================
   ```

---

## Dashboard Display

The dashboard will update automatically every 10 seconds:

```
Real-Time KPI Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣  Total Predictions:        2,847
    Shows: Total number of predictions processed

2️⃣  Max Confidence Score:    89.34% (woman)
    Shows: Highest probability prediction ever recorded

3️⃣  Lowest Confidence Score: 18.92% (background)
    Shows: Lowest confidence prediction on record

4️⃣  Average Confidence:      62.74%
    Shows: Average of all predictions

5️⃣  Most Frequent Q Type:    color
    Shows: Most common question category

Updated at: 2026-01-09T14:32:15
```

---

## Files Modified/Created

| File | Changes |
|------|---------|
| `main.py` | Added question categorization + Event Hub logging |
| `requirements.txt` | Added `azure-eventhub` package |
| `databricks_etl_pipeline.py` | ✨ NEW - ETL pipeline notebook |
| `databricks_kpi_dashboard.py` | ✨ NEW - KPI dashboard notebook |

---

## Monitoring & Troubleshooting

### Issue: No predictions appearing in dashboard
**Solution:**
1. Check Event Hub connection string in Databricks secrets
2. Verify Flask is logging (check container logs)
3. Make sure Databricks notebooks are running

### Issue: Dashboard shows 0 predictions
**Solution:**
1. Send a test prediction through the API
2. Wait 10 seconds for ETL to process
3. Refresh Databricks notebook

### Issue: "Event Hub not found"
**Solution:**
1. Verify Event Hub namespace and name exist in Azure
2. Check connection string format
3. Ensure connection string has read/send permissions

---

## Next Steps

Once dashboard is working:
1. Add alerting (Slack notification if avg confidence < 50%)
2. Add more granular filtering (by time period, model version, etc.)
3. Create SQL Warehouse queries for permanent storage
4. Set up Power BI dashboard on top of Delta Lake

---

## Cost Estimate

- **Event Hub**: $30-50/month (depends on throughput)
- **Databricks**: $0.50-1.00/hour (cluster running)
- **Total**: ~$100-200/month for continuous monitoring

---

## Questions?

Refer to this guide's sections for implementation details.
