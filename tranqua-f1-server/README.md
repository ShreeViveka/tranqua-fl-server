# Tranqua FL Server

Federated Learning aggregation server for Tranqua — Mental Health Tracker.

## What this does

- Receives anonymous model weight updates from users' laptops
- Runs FedAvg to combine them into an improved global model  
- Serves the improved model back to all clients
- **Never receives** diary entries, app usage, or personal data

## Deploy to Railway

1. Fork this repo
2. Connect to Railway → New Project → Deploy from GitHub
3. Railway auto-detects Procfile and deploys

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Server status |
| `GET /health` | Detailed health + round info |
| `POST /fl/upload` | Receive weight update from client |
| `GET /fl/model/weights` | Download current global model |
| `GET /fl/stats` | FL statistics |

## Privacy

Only model weight deltas (numbers) are received — never raw data.
Differential privacy noise is applied on the client before sending.
