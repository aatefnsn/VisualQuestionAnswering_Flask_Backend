# VQA Backend API Documentation

## Predict Endpoint

### Request
```
POST /predict
Content-Type: multipart/form-data

Parameters:
- file: Image file (jpg, jpeg, png) - Required
- question: Question string - Required

Example:
curl -X POST http://localhost:8080/predict \
  -F "file=@image.jpg" \
  -F "question=what color is the bus?"
```

### Response (200 OK)
```json
{
  "status": "success",
  "question_type": "color",
  "predicted_answers": [
    {
      "rank": 1,
      "class_id": 66,
      "class_name": "yellow",
      "probability": 0.5677,
      "confidence": "56.77%"
    },
    {
      "rank": 2,
      "class_id": 43,
      "class_name": "red",
      "probability": 0.2341,
      "confidence": "23.41%"
    },
    {
      "rank": 3,
      "class_id": 25,
      "class_name": "white",
      "probability": 0.1089,
      "confidence": "10.89%"
    },
    ...more predictions...
  ]
}
```

### Response Fields Explained

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "success" or "error" |
| `question_type` | string | Categorized question type: "color", "object", "count", "location", "action", "yes_no", "other" |
| `predicted_answers` | array | List of predictions sorted by confidence (highest first) |
| `rank` | integer | Position in ranking (1, 2, 3, ...) |
| `class_id` | integer | Internal model class identifier (0-999) |
| `class_name` | string | Human-readable answer (e.g., "yellow", "dog", "5") |
| `probability` | float | Raw probability (0.0 to 1.0) after softmax |
| `confidence` | string | Formatted percentage (e.g., "56.77%") |

### Question Type Categories
```
"color"     → Questions about colors (what color, what colors)
"object"    → Questions about objects (what, what is, what are)
"count"     → Questions about counting (how many, count)
"location"  → Questions about position (where, left, right, behind)
"action"    → Questions about actions (is, are, doing, wearing)
"yes_no"    → Yes/No questions (is there, are there, do, does)
"other"     → Questions that don't fit above categories
```

### Error Response (400/500)
```json
{
  "error": "error description",
  "details": "detailed error message",
  "trace": "full stack trace"
}
```

---

## What's New (Updated January 2026)

### Changes
1. **Softmax Normalization**: Probabilities now use softmax, resulting in proper 0-1 range (previously raw logits)
2. **Question Type Detection**: Every prediction now includes question category
3. **Better Formatting**: Confidence shown as percentage string (e.g., "56.77%")
4. **Event Hub Logging**: All predictions automatically logged to Azure Event Hub for analytics
5. **Top 20 Results**: Returns top 20 predictions instead of just 10

### Migration Guide for Frontend

**Old Response Format** (deprecated):
```json
{
  "class_name-0": "yellow",
  "class_name-1": "red",
  "class_name-2": "white"
}
```

**New Response Format**:
```json
{
  "status": "success",
  "question_type": "color",
  "predicted_answers": [
    {"rank": 1, "class_id": 66, "class_name": "yellow", "probability": 0.5677, "confidence": "56.77%"},
    {"rank": 2, "class_id": 43, "class_name": "red", "probability": 0.2341, "confidence": "23.41%"},
    {"rank": 3, "class_id": 25, "class_name": "white", "probability": 0.1089, "confidence": "10.89%"}
  ]
}
```

### Frontend Implementation Example

```javascript
// Fetch prediction
const formData = new FormData();
formData.append('file', imageFile);
formData.append('question', 'what color is the bus?');

const response = await fetch('/predict', {
  method: 'POST',
  body: formData
});

const data = await response.json();

// Display results
console.log(`Question Type: ${data.question_type}`);
console.log(`Top Answer: ${data.predicted_answers[0].class_name}`);
console.log(`Confidence: ${data.predicted_answers[0].confidence}`);

// Show top 3
data.predicted_answers.slice(0, 3).forEach(pred => {
  console.log(`${pred.rank}. ${pred.class_name}: ${pred.confidence}`);
});
```

---

## Metrics Automatically Logged

Every prediction is logged to Azure Event Hub with:
- Timestamp
- Question text
- Question type
- Top answer
- Top probability
- Model version
- User session ID

This data feeds into the **Real-Time KPI Dashboard** in Databricks.

---

## Health Check Endpoint

### Request
```
GET /health
```

### Response (200 OK)
```json
{
  "status": "healthy",
  "service": "vqa-backend"
}
```

---

## Rate Limiting
- **Limit**: 100 requests per day per IP address
- **Error**: 429 Too Many Requests

---

## Performance Notes
- Average response time: 2-3 seconds (includes image processing + model inference)
- Model size: 502 MB
- Memory requirement: 4 GB minimum
- GPU recommended for faster inference

---

## Version History
- **v1.0** (Jan 2026): Current version with softmax + Event Hub logging
- **v0.9** (Dec 2025): Initial release with raw logits
