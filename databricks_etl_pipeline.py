# Databricks notebook source
"""
VQA ETL Pipeline - Stream predictions from Event Hub to Delta Lake
This notebook must run continuously to feed data to the KPI dashboard
"""

# COMMAND ----------
# Cell 1: Setup
# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *

# Get Event Hub connection string from secrets
event_hub_connection_string = dbutils.secrets.get(scope="vqa", key="event_hub_connection_string")
event_hub_name = "vqa-predictions"

# Event Hub configuration
ehConf = {
  "eventhubs.connectionString": event_hub_connection_string,
  "eventhubs.consumerGroup": "$Default",
  "eventhubs.startingPosition": "-1",  # Start from latest
  "failOnDataLoss": "false"
}

print("✓ Event Hub configured")

# COMMAND ----------
# Cell 2: Read from Event Hub
# COMMAND ----------

df_raw = spark.readStream \
  .format("eventhubs") \
  .options(**ehConf) \
  .load()

# Parse JSON
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
    col("data.timestamp").cast(TimestampType()).alias("timestamp"),
    col("data.question"),
    col("data.question_type"),
    col("data.top_answer"),
    col("data.top_probability"),
    col("data.model_version"),
    col("data.user_session_id"),
    current_timestamp().alias("ingestion_time")
)

print("✓ Connected to Event Hub stream")

# COMMAND ----------
# Cell 3: Write to Delta Lake
# COMMAND ----------

predictions_delta_path = "/mnt/vqa/predictions"
checkpoint_path = "/tmp/vqa_predictions_checkpoint"

# Stream all predictions to Delta
query = df_predictions.writeStream \
    .format("delta") \
    .mode("append") \
    .option("checkpointLocation", checkpoint_path) \
    .option("path", predictions_delta_path) \
    .trigger(processingTime="5 seconds") \
    .start()

print("✓ Streaming predictions to Delta Lake")
print(f"✓ Delta path: {predictions_delta_path}")
print(f"✓ Checkpoint path: {checkpoint_path}")
print(f"✓ Updates every 5 seconds")

# COMMAND ----------
# Cell 4: Monitor Stream Status
# COMMAND ----------

while True:
    try:
        import time
        # Read latest predictions
        df_latest = spark.read.format("delta").load(predictions_delta_path)
        count = df_latest.count()
        
        # Show stats every 30 seconds
        if count > 0:
            stats = df_latest.select(
                count("*").alias("total_count"),
                avg("top_probability").alias("avg_confidence"),
                max("top_probability").alias("max_confidence"),
                min("top_probability").alias("min_confidence")
            ).collect()[0]
            
            print(f"[{datetime.now()}] Total: {count} | Avg: {stats.avg_confidence:.2%} | Max: {stats.max_confidence:.2%} | Min: {stats.min_confidence:.2%}")
        
        time.sleep(30)
        
    except KeyboardInterrupt:
        print("\n✓ Stopping stream monitoring")
        break

# COMMAND ----------
# To stop the streaming job, uncomment and run:
# query.stop()
# print("✓ Stream stopped")
