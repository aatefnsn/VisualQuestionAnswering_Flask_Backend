# Databricks notebook source
"""
VQA Real-Time KPI Dashboard
Tracks 5 Key Metrics:
1. Total prediction count
2. Max confidence answer (highest probability ever)
3. Lowest confidence prediction
4. Average confidence
5. Most frequent question type
"""

# COMMAND ----------
# Cell 1: Setup and Configuration
# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import datetime

# Get Event Hub connection string from secrets
event_hub_connection_string = dbutils.secrets.get(scope="vqa", key="event_hub_connection_string")
event_hub_name = "vqa-predictions"

# Event Hub configuration for Spark
ehConf = {
  "eventhubs.connectionString": event_hub_connection_string,
  "eventhubs.consumerGroup": "$Default",
  "eventhubs.startingPosition": "-1",  # Start from latest
  "failOnDataLoss": "false"
}

print("✓ Event Hub configured")
print(f"✓ Event Hub Name: {event_hub_name}")

# COMMAND ----------
# Cell 2: Read from Event Hub Stream
# COMMAND ----------

# Read predictions from Event Hub
df_raw = spark.readStream \
  .format("eventhubs") \
  .options(**ehConf) \
  .load()

# Parse JSON event data
df_predictions = df_raw.select(
    from_json(col("body").cast("string"), 
              StructType([
                  StructField("timestamp", StringType()),
                  StructField("question", StringType()),
                  StructField("question_type", StringType()),
                  StructField("top_answer", StringType()),
                  StructField("top_probability", DoubleType()),
                  StructField("model_version", StringType()),
                  StructField("user_session_id", StringType())
              ])).alias("data")
).select(
    col("data.timestamp").cast(TimestampType()),
    col("data.question"),
    col("data.question_type"),
    col("data.top_answer"),
    col("data.top_probability"),
    col("data.model_version"),
    col("data.user_session_id")
)

print("✓ Connected to Event Hub stream")

# COMMAND ----------
# Cell 3: Calculate Real-Time KPIs
# COMMAND ----------

# Calculate all 5 KPIs in a single aggregation
kpi_stats = df_predictions.groupBy().agg(
    # KPI 1: Total count of predictions
    count("*").alias("total_predictions"),
    
    # KPI 2: Max confidence (highest probability ever)
    max("top_probability").alias("max_confidence"),
    
    # KPI 3: Lowest confidence (minimum probability)
    min("top_probability").alias("min_confidence"),
    
    # KPI 4: Average confidence
    round(avg("top_probability"), 4).alias("avg_confidence"),
    
    # KPI 5: Most frequent question type
    mode("question_type").alias("most_frequent_question_type"),
    
    # Bonus: Get the answer for max confidence
    first(when(col("top_probability") == max("top_probability").over(), col("top_answer"))).alias("highest_confidence_answer"),
    
    # Bonus: Get the answer for min confidence
    first(when(col("top_probability") == min("top_probability").over(), col("top_answer"))).alias("lowest_confidence_answer"),
    
    # Timestamp
    current_timestamp().alias("dashboard_updated_at")
)

# COMMAND ----------
# Cell 4: Write KPIs to Delta Lake (Real-Time Updates)
# COMMAND ----------

# Path for KPI metrics
kpi_delta_path = "/mnt/vqa/kpi_metrics"
kpi_checkpoint = "/tmp/vqa_kpi_checkpoint"

# Write streaming aggregation to Delta Lake
query = kpi_stats.writeStream \
    .format("delta") \
    .mode("overwrite") \
    .option("checkpointLocation", kpi_checkpoint) \
    .option("path", kpi_delta_path) \
    .trigger(processingTime="10 seconds") \
    .start()

print("✓ Streaming KPI metrics to Delta Lake")
print(f"✓ Updating every 10 seconds")
print(f"✓ Delta path: {kpi_delta_path}")

# COMMAND ----------
# Cell 5: Display Live KPI Dashboard
# COMMAND ----------

# Read current KPI values
kpi_current = spark.read.format("delta").load(kpi_delta_path).select(
    "total_predictions",
    "max_confidence",
    "min_confidence",
    "avg_confidence",
    "most_frequent_question_type",
    "highest_confidence_answer",
    "lowest_confidence_answer",
    "dashboard_updated_at"
)

# Convert to Pandas for nice display
import pandas as pd
kpi_df = kpi_current.toPandas()

if len(kpi_df) > 0:
    row = kpi_df.iloc[0]
    
    print("=" * 60)
    print("VQA REAL-TIME KPI DASHBOARD")
    print("=" * 60)
    print(f"✓ Last Updated: {row['dashboard_updated_at']}")
    print()
    print("📊 KEY METRICS:")
    print(f"  1️⃣  Total Predictions:        {int(row['total_predictions']):,}")
    print(f"  2️⃣  Max Confidence Score:    {row['max_confidence']*100:.2f}% ({row['highest_confidence_answer']})")
    print(f"  3️⃣  Lowest Confidence Score: {row['min_confidence']*100:.2f}% ({row['lowest_confidence_answer']})")
    print(f"  4️⃣  Average Confidence:      {row['avg_confidence']*100:.2f}%")
    print(f"  5️⃣  Most Frequent Q Type:    {row['most_frequent_question_type']}")
    print("=" * 60)
    
    display(kpi_current)
else:
    print("⏳ Waiting for first predictions...")

# COMMAND ----------
# Cell 6: Query Historical Trend (Optional)
# COMMAND ----------

# Read all predictions from Delta
df_all_predictions = spark.read.format("delta").load("/mnt/vqa/predictions")

# Show last 10 predictions
last_10 = df_all_predictions.orderBy(desc("timestamp")).limit(10).select(
    "timestamp",
    "question",
    "question_type",
    "top_answer",
    round(col("top_probability") * 100, 2).alias("confidence_%"),
    "model_version"
)

print("\n📈 LAST 10 PREDICTIONS:")
display(last_10)

# COMMAND ----------
# Cell 7: Stop streaming (Optional - use to stop the notebook)
# COMMAND ----------

# To stop streaming, run this cell:
# query.stop()
# print("✓ Streaming stopped")
