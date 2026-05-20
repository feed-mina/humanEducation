
       GCP Free-Tier Deployment Plan for K-Ride MSA

       Table of Contents

       1. Current State Summary
       2. Phase 1 -- Free Tier / Minimal Cost
       3. Phase 2 -- With GCP Credits (GPU)
       4. Phase 3 -- Production Pipeline
       5. CI/CD Migration
       6. Code Changes Required
       7. Risk Matrix

       ---
       0. Current State Summary

       Services inventory from codebase analysis:

       ┌─────────────────┬──────┬────────────────────────────┬─────────────────────────────────────────────┬────────────────────────────────────────
       ───┐
       │     Service     │ Port │          Runtime           │                   Docker                    │             Key Dependencies
          │
       ├─────────────────┼──────┼────────────────────────────┼─────────────────────────────────────────────┼────────────────────────────────────────
       ───┤
       │ Spring Boot     │      │                            │ subproject/SDUI/SDUI-server/Dockerfile      │ PostgreSQL, Redis, Flyway, JWT,
          │
       │ (SDUI)          │ 8080 │ Java 17, Temurin           │ (multi-stage)                               │ Kakao/Google OAuth, AWS S3, GCP
       Document  │
       │                 │      │                            │                                             │ AI
          │
       ├─────────────────┼──────┼────────────────────────────┼─────────────────────────────────────────────┼────────────────────────────────────────
       ───┤
       │ FastAPI         │ 8000 │ Python 3.9, torch 2.11     │ subproject/ml-server/Dockerfile             │ Neo4j AuraDB, ChromaDB (local), Groq
       LLM, │
       │ (K-Ride)        │      │                            │                                             │  Supabase, NetworkX graph, pickle
       models  │
       ├─────────────────┼──────┼────────────────────────────┼─────────────────────────────────────────────┼────────────────────────────────────────
       ───┤
       │ NLP Chatbot     │ 8001 │ Python 3.9+,               │ No Dockerfile yet                           │ ChromaDB (local, shared), Groq LLM,
          │
       │                 │      │ sentence-transformers      │                                             │ cross-encoder reranker
          │
       ├─────────────────┼──────┼────────────────────────────┼─────────────────────────────────────────────┼────────────────────────────────────────
       ───┤
       │ Next.js PWA     │ 3000 │ Node.js, React 19          │ None (Vercel)                               │ Proxies to 8080 + 8000
          │
       ├─────────────────┼──────┼────────────────────────────┼─────────────────────────────────────────────┼────────────────────────────────────────
       ───┤
       │ team/model      │ 8000 │ Python 3.11                │ team/model/Dockerfile                       │ PaddleOCR, ultralytics
          │
       │ (PaddleOCR)     │      │                            │                                             │
          │
       └─────────────────┴──────┴────────────────────────────┴─────────────────────────────────────────────┴────────────────────────────────────────
       ───┘









       External cloud services (no migration needed):
       - Neo4j AuraDB -- already cloud-hosted
       - Supabase -- already cloud-hosted (images + data)
       - Groq LLM API -- external API
       - Vercel -- frontend hosting

       Data volumes that need migration:
       - ChromaDB local directory (chroma_db/) -- ~5 collections, used by both FastAPI (port 8000) and Chatbot (port 8001)
       - PostgreSQL (SDUI) on EC2 Docker -- sdui-db container
       - Model files: models/dl/weather_lstm.pt, models/safety_regressor.pkl, models/safety_classifier.pkl, route_graph.pkl, ensemble ranker pickle
       - CSV data files: road_scored.csv, facility_clean.csv, tour_poi.csv
       - PDF dataset: 17+ files in dataset/pdf/

       Current CI/CD (from D:\kride-project\subproject\SDUI\.github\workflows\deploy.yml):
       - GitHub Actions -> Docker Hub -> SSH to AWS EC2
       - Spring Boot only (FastAPI deployment commented out)
       - Uses appleboy/ssh-action for remote Docker commands

       ---
       Phase 1: Free Tier / Minimal Cost ($0-$15/month)

       Goal

       Get all services running on GCP using free tier and near-free resources. Sufficient for development and demo traffic (< 100 users/day).

       1.1 Service Placement

       ┌──────────────┬────────────────────────┬────────────────────────────────────────────────────────────────────────────────┬───────────────────
       ────┐
       │   Service    │      GCP Resource      │                                   Rationale                                    │     Monthly Cost
           │
       ├──────────────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────────
       ────┤
       │ Spring Boot  │ Cloud Run (free tier)  │ Stateless JAR, scales to zero. Free tier: 2M requests, 360K vCPU-seconds, 180K │ $0
           │
       │ (SDUI)       │                        │  GiB-seconds                                                                   │
           │
       ├──────────────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────────
       ────┤
       │ FastAPI      │ Compute Engine         │ Loads ~200MB of models + NetworkX graph into memory at startup. Cloud Run cold │
           │
       │ (K-Ride)     │ e2-micro               │  start would be 30-60s. Always-on is needed. Free tier: 1 e2-micro in          │ $0
           │
       │              │                        │ us-central1, 30GB standard disk                                                │
           │
       ├──────────────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────────
       ────┤
       │ NLP Chatbot  │ Colocated on same      │ Shares ChromaDB with FastAPI. sentence-transformers model is ~100MB            │ $0 (shared)
           │
       │              │ e2-micro               │                                                                                │
           │
       ├──────────────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────────
       ────┤
       │ PostgreSQL   │ Cloud SQL (free trial) │ Cloud SQL has 90-day free trial. After trial: use Supabase free tier (500MB)   │ $0 (trial) -> $0
           │
       │ (SDUI)       │  or Supabase free tier │ or Neon free tier (you already have a Neon URL in .env)                        │ (Neon/Supabase)
           │
       ├──────────────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────────
       ────┤
       │ Redis        │ Memorystore free trial │ Spring Boot uses Redis for cache + SQL query cache. Alternative: use Cloud     │ $0
           │
       │              │  or Cloud Run sidecar  │ Run's built-in memory for small cache, or Redis on e2-micro                    │
           │
       ├──────────────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────────
       ────┤
       │ ChromaDB     │ Local on e2-micro      │ Both FastAPI and Chatbot read from chroma_db/. Keep as local directory on the  │ $0
           │
       │              │                        │ e2-micro instance. Total data is small (< 500MB)                               │
           │
       ├──────────────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────────
       ────┤
       │ Next.js PWA  │ Vercel (stay)          │ Already deployed. No change needed                                             │ $0
           │
       ├──────────────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────────
       ────┤
       │ Model files  │ Cloud Storage (5GB     │ Store .pt, .pkl, CSV files. Download to e2-micro at startup                    │ $0
           │
       │              │ free)                  │                                                                                │
           │
       └──────────────┴────────────────────────┴────────────────────────────────────────────────────────────────────────────────┴───────────────────
       ────┘


















       1.2 Architecture Diagram (Phase 1)

       [Vercel - Next.js PWA]
               |
               |-- /api/* --> [Cloud Run - Spring Boot :8080]
               |                    |
               |                    |--> Cloud SQL / Neon (PostgreSQL)
               |                    |--> Redis (Memorystore / in-process)
               |                    |--> Supabase (external)
               |
               |-- /kride-api/* --> [e2-micro VM - FastAPI :8000 + Chatbot :8001]
                                         |
                                         |--> ChromaDB (local dir)
                                         |--> Neo4j AuraDB (external)
                                         |--> Groq LLM (external)
                                         |--> GCS (model files)

       1.3 Compute Engine e2-micro Setup

       The e2-micro instance (0.25 vCPU, 1GB RAM) is tight for running both FastAPI + Chatbot with ML models. Critical optimizations:

       Memory budget on 1GB RAM:
       - OS overhead: ~200MB
       - FastAPI + NetworkX graph + pickle models: ~300MB
       - Chatbot + sentence-transformers (multilingual-e5-small, 384-dim): ~200MB
       - ChromaDB: ~100MB
       - Remaining: ~200MB

       This is extremely tight. If it does not fit, upgrade to e2-small (2GB RAM, ~$6.11/month in us-central1) or e2-medium (4GB, ~$12.22/month).

       Recommendation: Start with e2-small ($6.11/month) for reliability. The e2-micro free tier only has 1GB which will likely OOM when loading
       sentence-transformers + torch models simultaneously.

       Docker Compose on e2-micro/e2-small:

       # docker-compose.gcp.yml (to be created)
       version: '3.8'
       services:
         kride-fastapi:
           build:
             context: .
             dockerfile: Dockerfile.fastapi  # new, trimmed requirements
           ports:
             - "8000:8000"
           volumes:
             - ./chroma_db:/app/chroma_db
             - ./models:/app/models
             - ./data:/app/data
           env_file: .env.gcp

         kride-chatbot:
           build:
             context: ./subproject/NLP
             dockerfile: Dockerfile.chatbot  # new
           ports:
             - "8001:8001"
           volumes:
             - ./chroma_db:/app/chroma_db  # shared ChromaDB
           env_file: .env.gcp

         redis:
           image: redis:7-alpine
           ports:
             - "6379:6379"

       1.4 Cloud Run for Spring Boot

       The existing subproject/SDUI/SDUI-server/Dockerfile is reusable with minimal changes.

       Cloud Run configuration:
       - Region: asia-northeast3 (Seoul) for latency, or us-central1 for free tier eligibility
       - Memory: 512MB (Spring Boot JAR is ~150MB, runtime ~400MB)
       - CPU: 1 vCPU
       - Min instances: 0 (scale to zero for cost savings)
       - Max instances: 2
       - Concurrency: 80

       Critical issue: Cloud Run requires the database connection to go through a Cloud SQL Auth Proxy or a public IP. If using Neon/Supabase for
       PostgreSQL, this is simpler since they provide public connection strings with SSL.

       1.5 Database Strategy

       Option A (recommended for Phase 1): Use Neon free tier
       - You already have a Neon connection string in .env:
       postgresql://neondb_owner:npg_...@ep-fragrant-dream-aozv5jxk-pooler.c-2.ap-southeast-1.aws.neon.tech/neondb
       - Neon free tier: 0.5GB storage, auto-suspend after 5 minutes of inactivity
       - Run Flyway migrations against Neon from local machine first
       - Change SPRING_DATASOURCE_URL in Cloud Run env vars

       Option B: Cloud SQL free trial (90 days)
       - db-f1-micro instance, 10GB storage
       - After 90 days, costs ~$7.67/month
       - Better performance than Neon for sustained workloads

       Redis:
       - For Phase 1, embed Redis in the e2-micro/e2-small VM via Docker
       - Spring Boot on Cloud Run connects to the VM's Redis via internal VPC or public IP with firewall rules
       - Alternative: skip Redis entirely and disable caching in Spring Boot for Phase 1 (set spring.data.redis.host to empty, rely on in-memory
       cache)

       1.6 Networking

       VPC: default
       Subnet: us-central1 (or asia-northeast3)

       Cloud Run (Spring Boot)
         ├── Ingress: Allow all (public HTTPS)
         ├── Egress: connects to Neon PG (external), Redis on VM (internal VPC)
         └── Custom domain: yerin.duckdns.org (or new GCP domain)

       e2-micro/e2-small VM
         ├── External IP: Ephemeral (or static for $0 if attached to running instance)
         ├── Firewall: allow TCP 8000, 8001 from Vercel IPs + Cloud Run
         └── Internal: Redis on localhost:6379

       Firewall rules to create:
       - allow-kride-api: TCP 8000, 8001 from 0.0.0.0/0 (or restrict to Vercel/Cloud Run egress IPs)
       - allow-ssh: TCP 22 from your IP only

       1.7 Data Migration Steps

       1. PostgreSQL (SDUI): Export from EC2 sdui-db container via pg_dump, import to Neon/Cloud SQL via psql. Flyway will validate migrations.
       2. ChromaDB: Tar the chroma_db/ directory, upload to GCS, download to e2-micro.
       3. Model files: Upload models/ directory to GCS bucket. On VM startup, gsutil cp gs://kride-models/* /app/models/.
       4. CSV data: Same as models -- upload to GCS, download on startup.
       5. PDF dataset: Upload to GCS. Chatbot can read from mounted GCS FUSE or download at startup.

       1.8 Cost Summary (Phase 1)

       ┌─────────────────────────────────┬─────────────────┐
       │            Resource             │  Monthly Cost   │
       ├─────────────────────────────────┼─────────────────┤
       │ Cloud Run (Spring Boot)         │ $0 (free tier)  │
       ├─────────────────────────────────┼─────────────────┤
       │ e2-small VM (FastAPI + Chatbot) │ $6.11           │
       ├─────────────────────────────────┼─────────────────┤
       │ Neon PostgreSQL                 │ $0 (free tier)  │
       ├─────────────────────────────────┼─────────────────┤
       │ Redis (on VM)                   │ $0 (shared)     │
       ├─────────────────────────────────┼─────────────────┤
       │ Cloud Storage (models + data)   │ $0 (< 5GB free) │
       ├─────────────────────────────────┼─────────────────┤
       │ Vercel (Next.js)                │ $0 (free tier)  │
       ├─────────────────────────────────┼─────────────────┤
       │ Neo4j AuraDB                    │ $0 (existing)   │
       ├─────────────────────────────────┼─────────────────┤
       │ Supabase                        │ $0 (existing)   │
       ├─────────────────────────────────┼─────────────────┤
       │ Total                           │ ~$6/month       │
       └─────────────────────────────────┴─────────────────┘

       ---
       Phase 2: With GCP Credits -- Add GPU for TorchServe ($150-$400/month)

       Goal

       Add GPU-backed TorchServe for current ML models + future media pipeline. Use GCP credits to offset costs.

       2.1 Service Placement Changes

       ┌───────────────────────┬──────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┐
       │        Service        │               GCP Resource               │                          Change from Phase 1                          │
       ├───────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
       │ Spring Boot           │ Cloud Run                                │ No change                                                             │
       ├───────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
       │ FastAPI (K-Ride)      │ Cloud Run (migrate from VM)              │ Move to Cloud Run now that GPU VM handles models                      │
       ├───────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
       │ NLP Chatbot           │ Cloud Run                                │ Move to Cloud Run, ChromaDB moves to persistent disk on GPU VM or GCS │
       ├───────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
       │ TorchServe            │ Compute Engine g2-standard-4 (1x L4 GPU) │ NEW                                                                   │
       ├───────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
       │ Celery Worker         │ Colocated on GPU VM                      │ NEW                                                                   │
       ├───────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
       │ Redis (Celery broker) │ Memorystore for Redis (1GB)              │ Upgrade from VM-local                                                 │
       └───────────────────────┴──────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────┘

       2.2 GPU Instance Selection

       Recommended: g2-standard-4 with 1x NVIDIA L4 (Spot VM)

       ┌─────────────────┬───────────────────────────┐
       │      Spec       │           Value           │
       ├─────────────────┼───────────────────────────┤
       │ vCPUs           │ 4                         │
       ├─────────────────┼───────────────────────────┤
       │ RAM             │ 16 GB                     │
       ├─────────────────┼───────────────────────────┤
       │ GPU             │ NVIDIA L4 (24GB VRAM)     │
       ├─────────────────┼───────────────────────────┤
       │ On-demand price │ ~$0.70/hour = ~$504/month │
       ├─────────────────┼───────────────────────────┤
       │ Spot price      │ ~$0.21/hour = ~$151/month │
       └─────────────────┴───────────────────────────┘

       Alternative: n1-standard-4 + 1x T4 (Spot VM)

       ┌────────────┬───────────────────────────┐
       │    Spec    │           Value           │
       ├────────────┼───────────────────────────┤
       │ vCPUs      │ 4                         │
       ├────────────┼───────────────────────────┤
       │ RAM        │ 15 GB                     │
       ├────────────┼───────────────────────────┤
       │ GPU        │ NVIDIA T4 (16GB VRAM)     │
       ├────────────┼───────────────────────────┤
       │ Spot price │ ~$0.14/hour = ~$101/month │
       └────────────┴───────────────────────────┘

       Recommendation: Use T4 Spot for Phase 2 (cheaper, sufficient for current models). L4 for Phase 3 when CogVideoX is needed.

       2.3 TorchServe Model Packaging

       Current models to package as .mar files:

       ┌─────────────────────────────────────────────────┬────────┬──────────────────────────────────────────────────┐
       │                      Model                      │  Size  │                TorchServe Handler                │
       ├─────────────────────────────────────────────────┼────────┼──────────────────────────────────────────────────┤
       │ WeatherLSTM (weather_lstm.pt)                   │ ~5MB   │ Custom handler: load LSTM, predict weather class │
       ├─────────────────────────────────────────────────┼────────┼──────────────────────────────────────────────────┤
       │ Safety Regressor (safety_regressor.pkl)         │ ~10MB  │ Custom handler: sklearn predict                  │
       ├─────────────────────────────────────────────────┼────────┼──────────────────────────────────────────────────┤
       │ Safety Classifier (safety_classifier.pkl)       │ ~10MB  │ Custom handler: sklearn predict                  │
       ├─────────────────────────────────────────────────┼────────┼──────────────────────────────────────────────────┤
       │ Ensemble Ranker (LightGBM/XGBoost)              │ ~20MB  │ Custom handler: load pickle, rank POIs           │
       ├─────────────────────────────────────────────────┼────────┼──────────────────────────────────────────────────┤
       │ Cross-encoder reranker (ms-marco-MiniLM-L-6-v2) │ ~80MB  │ Default text handler                             │
       ├─────────────────────────────────────────────────┼────────┼──────────────────────────────────────────────────┤
       │ Sentence-transformers (multilingual-e5-small)   │ ~100MB │ Default text handler                             │
       └─────────────────────────────────────────────────┴────────┴──────────────────────────────────────────────────┘

       TorchServe config (config.properties):
       inference_address=http://0.0.0.0:8085
       management_address=http://0.0.0.0:8086
       grpc_inference_port=7070
       model_store=/home/model-store
       load_models=all
       number_of_gpu=1

       2.4 Celery + Redis Architecture

       [Cloud Run - FastAPI] --(task submit)--> [Memorystore Redis]
                                                     |
                                           [GPU VM - Celery Worker]
                                                     |
                                           [GPU VM - TorchServe :8085]

       New files needed:
       - src/api/celery_app.py -- Celery application config
       - src/api/tasks.py -- Celery task definitions (weather prediction, ensemble ranking, reranking)
       - torchserve/ directory with model handlers and MAR packaging scripts

       2.5 ChromaDB Migration

       Move ChromaDB from local filesystem to a shared solution:

       Option A: Keep on persistent disk attached to GPU VM
       - FastAPI and Chatbot on Cloud Run access ChromaDB via a thin HTTP wrapper (chroma-server mode)
       - Run chroma run --path /data/chroma_db --host 0.0.0.0 --port 8100 on the GPU VM

       Option B: Use Chroma Cloud (managed service)
       - Chroma offers a hosted solution; may have cost implications

       Recommendation: Option A. Run ChromaDB server mode on the GPU VM alongside TorchServe. The GPU VM is always-on anyway.

       2.6 Cost Summary (Phase 2)

       ┌────────────────────────────────────────────────┬──────────────────┐
       │                    Resource                    │   Monthly Cost   │
       ├────────────────────────────────────────────────┼──────────────────┤
       │ Cloud Run (Spring Boot + FastAPI + Chatbot)    │ $0-$5            │
       ├────────────────────────────────────────────────┼──────────────────┤
       │ Spot VM n1+T4 (TorchServe + Celery + ChromaDB) │ ~$101            │
       ├────────────────────────────────────────────────┼──────────────────┤
       │ Memorystore Redis (1GB)                        │ ~$35             │
       ├────────────────────────────────────────────────┼──────────────────┤
       │ Neon / Cloud SQL PostgreSQL                    │ $0-$8            │
       ├────────────────────────────────────────────────┼──────────────────┤
       │ Cloud Storage                                  │ $0               │
       ├────────────────────────────────────────────────┼──────────────────┤
       │ Total                                          │ ~$140-$150/month │
       └────────────────────────────────────────────────┴──────────────────┘

       With GCP credits, this is effectively $0 out-of-pocket.

       ---
       Phase 3: Production -- Full Media Pipeline ($400-$700/month)

       Goal

       Add Photo -> TTS -> Video generation pipeline with Track A (drawings) and Track B (photos).

       3.1 Full Architecture

       [Vercel - Next.js PWA]
               |
               |-- /api/* --> [Cloud Run - Spring Boot]
               |                    |
               |                    |--> Cloud SQL (PostgreSQL)
               |                    |--> Memorystore (Redis)
               |                    |--> Supabase
               |
               |-- /kride-api/* --> [Cloud Run - FastAPI Orchestrator]
                                         |
                                         |--(sync)--> Cloud Run - Chatbot
                                         |
                                         |--(async celery task)--> Memorystore Redis
                                                                       |
                                         [GPU VM - Celery Worker Pool]
                                              |
                                              |--> TorchServe :8085
                                              |      |- WeatherLSTM
                                              |      |- Ensemble Ranker
                                              |      |- Cross-encoder
                                              |      |- Sentence-transformers
                                              |      |- GPT-SoVITS (TTS)
                                              |      |- Animated Drawings
                                              |      |- CogVideoX / 3D Photo Inpainting
                                              |      |- BLIP-2 (image captioning)
                                              |
                                              |--> ChromaDB server :8100
                                              |
                                              |--> GCS (upload final videos)
                                              |
                                              |--> FCM (push notification to PWA)

       3.2 GPU VM Upgrade

       For Phase 3, CogVideoX and GPT-SoVITS require more VRAM:

       Recommended: g2-standard-8 with 1x NVIDIA L4 (Spot)

       ┌────────────┬───────────────────────────┐
       │    Spec    │           Value           │
       ├────────────┼───────────────────────────┤
       │ vCPUs      │ 8                         │
       ├────────────┼───────────────────────────┤
       │ RAM        │ 32 GB                     │
       ├────────────┼───────────────────────────┤
       │ GPU        │ 1x L4 (24GB VRAM)         │
       ├────────────┼───────────────────────────┤
       │ Spot price │ ~$0.42/hour = ~$302/month │
       └────────────┴───────────────────────────┘

       If CogVideoX needs >24GB VRAM, use a2-highgpu-1g with 1x A100 40GB ($1.10/hour Spot = ~$792/month) -- but this is expensive even with
       credits. Consider using g2-standard-8 and running CogVideoX with model quantization (INT8) to fit in 24GB.

       3.3 TorchServe Model Additions (Phase 3)

       ┌──────────────────────┬──────────┬─────────────────────────────────────────────┐
       │        Model         │   VRAM   │                  Container                  │
       ├──────────────────────┼──────────┼─────────────────────────────────────────────┤
       │ GPT-SoVITS (TTS)     │ ~4GB     │ Separate Docker container with Python 3.10+ │
       ├──────────────────────┼──────────┼─────────────────────────────────────────────┤
       │ Animated Drawings    │ ~2GB     │ Separate container (Python 3.8, CUDA 11)    │
       ├──────────────────────┼──────────┼─────────────────────────────────────────────┤
       │ CogVideoX 1.5 (INT8) │ ~12-16GB │ Separate container (Python 3.10, CUDA 12)   │
       ├──────────────────────┼──────────┼─────────────────────────────────────────────┤
       │ 3D Photo Inpainting  │ ~4GB     │ Separate container                          │
       ├──────────────────────┼──────────┼─────────────────────────────────────────────┤
       │ BLIP-2               │ ~4GB     │ Shared with CogVideoX container             │
       ├──────────────────────┼──────────┼─────────────────────────────────────────────┤
       │ MusicGen (BGM)       │ ~4GB     │ Optional, lower priority                    │
       └──────────────────────┴──────────┴─────────────────────────────────────────────┘

       Key insight from architecture_expand_0520.md: Different models require different Python/CUDA versions. TorchServe solves this via .mar
       packaging, but for models that fundamentally conflict (Python 3.8 vs 3.10, CUDA 11 vs 12), use separate Docker containers with their own
       TorchServe instances.

       # docker-compose.gpu.yml
       services:
         torchserve-main:  # WeatherLSTM, ensemble, reranker, embeddings
           image: pytorch/torchserve:latest-gpu
           deploy:
             resources:
               reservations:
                 devices:
                   - capabilities: [gpu]
           ports:
             - "8085:8085"
             - "8086:8086"

         torchserve-tts:  # GPT-SoVITS
           build: ./torchserve/tts
           deploy:
             resources:
               reservations:
                 devices:
                   - capabilities: [gpu]
           ports:
             - "8087:8085"

         torchserve-video:  # CogVideoX, 3D Photo Inpainting, BLIP-2
           build: ./torchserve/video
           deploy:
             resources:
               reservations:
                 devices:
                   - capabilities: [gpu]
           ports:
             - "8088:8085"

         celery-worker:
           build: ./src/api
           command: celery -A celery_app worker -l info -c 2
           depends_on:
             - redis
             - torchserve-main
           env_file: .env.gcp

         chromadb:
           image: chromadb/chroma:latest
           ports:
             - "8100:8000"
           volumes:
             - chroma-data:/chroma/chroma

         redis:
           image: redis:7-alpine
           ports:
             - "6379:6379"

       3.4 Celery Task Flow (Media Pipeline)

       # src/api/tasks.py (new file)
       from celery import chain, group
       from celery_app import celery

       @celery.task(bind=True, max_retries=3)
       def classify_input(self, image_url: str, text: str):
           """Determine Track A (drawing) or Track B (photo)"""
           # Call BLIP-2 on TorchServe
           # Return: {"track": "A" | "B", "caption": "...", "prompt": "..."}

       @celery.task(bind=True)
       def generate_tts(self, text: str, voice_id: str):
           """GPT-SoVITS TTS generation"""
           # Call torchserve-tts:8087

       @celery.task(bind=True)
       def generate_video_track_a(self, image_url: str, motion: str):
           """Animated Drawings pipeline"""

       @celery.task(bind=True)
       def generate_video_track_b(self, image_url: str, prompt: str):
           """CogVideoX or 3D Photo Inpainting"""

       @celery.task(bind=True)
       def mix_final(self, video_path: str, audio_path: str, bgm_path: str):
           """FFmpeg mixing -> upload to GCS -> FCM notification"""

       3.5 Cost Summary (Phase 3)

       ┌──────────────────────────────┬──────────────────┐
       │           Resource           │   Monthly Cost   │
       ├──────────────────────────────┼──────────────────┤
       │ Cloud Run (3 services)       │ $5-$20           │
       ├──────────────────────────────┼──────────────────┤
       │ Spot VM g2-standard-8 + L4   │ ~$302            │
       ├──────────────────────────────┼──────────────────┤
       │ Memorystore Redis (2GB)      │ ~$55             │
       ├──────────────────────────────┼──────────────────┤
       │ Cloud SQL (db-g1-small)      │ ~$26             │
       ├──────────────────────────────┼──────────────────┤
       │ Cloud Storage (videos, 50GB) │ ~$1              │
       ├──────────────────────────────┼──────────────────┤
       │ Total                        │ ~$400-$410/month │
       └──────────────────────────────┴──────────────────┘

       ---
       CI/CD Pipeline Migration

       Current Pipeline (from D:\kride-project\subproject\SDUI\.github\workflows\deploy.yml)

       GitHub push -> Build JAR -> Docker build -> Docker Hub push -> SSH to EC2 -> docker run

       New Pipeline (GCP)

       Spring Boot -> Cloud Run:
       # .github/workflows/deploy-gcp.yml
       name: Deploy to GCP Cloud Run
       on:
         push:
           branches: [main]
       jobs:
         deploy-spring-boot:
           runs-on: ubuntu-latest
           steps:
             - uses: actions/checkout@v4
             - uses: google-github-actions/auth@v2
               with:
                 credentials_json: ${{ secrets.GCP_SA_KEY }}
             - uses: google-github-actions/setup-gcloud@v2
             - name: Build and push to Artifact Registry
               run: |
                 gcloud builds submit ./subproject/SDUI/SDUI-server \
                   --tag asia-northeast3-docker.pkg.dev/$PROJECT_ID/kride/sdui-backend:${{ github.sha }}
             - name: Deploy to Cloud Run
               run: |
                 gcloud run deploy sdui-backend \
                   --image asia-northeast3-docker.pkg.dev/$PROJECT_ID/kride/sdui-backend:${{ github.sha }} \
                   --region asia-northeast3 \
                   --memory 512Mi \
                   --cpu 1 \
                   --min-instances 0 \
                   --max-instances 2 \
                   --set-env-vars "SPRING_PROFILES_ACTIVE=prod,SPRING_DATASOURCE_URL=${{ secrets.GCP_DB_URL }}"

       FastAPI -> Compute Engine (Phase 1) / Cloud Run (Phase 2+):
         deploy-fastapi:
           runs-on: ubuntu-latest
           steps:
             # Phase 1: SSH to e2-micro, pull and restart
             # Phase 2+: Deploy to Cloud Run similar to Spring Boot

       ---
       Code Changes Required

       Phase 1 Changes

       1. Create Dockerfile.chatbot for NLP Chatbot (new file at subproject/NLP/Dockerfile.chatbot)
       FROM python:3.9-slim
       WORKDIR /app
       COPY chatbot/requirements.txt .
       RUN pip install --no-cache-dir -r requirements.txt
       COPY chatbot/ ./chatbot/
       CMD ["uvicorn", "chatbot.chatbot_server:app", "--host", "0.0.0.0", "--port", "8001"]

       2. Update next.config.ts (line 12, 29)
       - Change BACKEND_URL to Cloud Run URL
       - Change /kride-api/* destination from localhost:8000 to the VM's external IP or Cloud Run URL
       const BACKEND_URL = isProd
           ? 'https://sdui-backend-xxxxx-an.a.run.app'  // Cloud Run URL
           : 'http://localhost:8080';
       // ...
       { source: '/kride-api/:path*', destination: isProd ? 'http://VM_IP:8000/api/:path*' : 'http://localhost:8000/api/:path*' }

       3. Update CSP connect-src in next.config.ts (line 61)
       - Add Cloud Run and VM URLs to allowed connections

       4. Create docker-compose.gcp.yml (new file at project root)
       - For running FastAPI + Chatbot + Redis on the VM

       5. Update .env to create .env.gcp variant
       - DATABASE_URL -> Neon connection string
       - Add CHROMA_PATH=/app/chroma_db
       - Keep all external API keys (Neo4j, Groq, Supabase, Kakao)

       6. Spring Boot application.yml -- create application-prod.yml profile
       - spring.datasource.url -> Neon/Cloud SQL URL with SSL
       - spring.data.redis.host -> VM internal IP or Memorystore IP
       - kride.fastapi.url -> http://VM_IP:8000
       - kride.fastapi.chatbot-url -> http://VM_IP:8001

       Phase 2 Changes

       7. Create Celery infrastructure:
       - src/api/celery_app.py -- Celery app definition with Redis broker
       - src/api/tasks.py -- Task definitions for model inference
       - Modify src/api/fastapi_server.py to submit tasks to Celery instead of running inference inline

       8. Create TorchServe handlers:
       - torchserve/weather_lstm_handler.py
       - torchserve/ensemble_handler.py
       - torchserve/reranker_handler.py
       - MAR packaging script: torchserve/package.sh

       9. ChromaDB client mode: Update subproject/NLP/chatbot/config.py line 10:
       # From:
       CHROMA_PATH = str(PROJECT_ROOT / "chroma_db")
       # To:
       CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
       CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8100"))

       Phase 3 Changes

       10. Media pipeline tasks: New Celery tasks for Track A/B
       11. FCM integration: Push notifications when video is ready
       12. GCS upload utility: Upload generated videos to Cloud Storage
       13. WebSocket/SSE progress endpoint: Real-time progress updates to PWA

       ---
       Risk Matrix

       ┌──────────────────────────────────────┬──────────────────────────┬──────────────────────────────────────────────────────────────────────────
       ┐
       │                 Risk                 │          Impact          │                                Mitigation
       │
       ├──────────────────────────────────────┼──────────────────────────┼──────────────────────────────────────────────────────────────────────────
       ┤
       │ e2-micro OOM (1GB RAM)               │ FastAPI/Chatbot crash    │ Use e2-small ($6/month) or split into 2 Cloud Run services
       │
       ├──────────────────────────────────────┼──────────────────────────┼──────────────────────────────────────────────────────────────────────────
       ┤
       │ Spot VM preemption                   │ GPU tasks interrupted    │ Celery auto-retry with acks_late=True; tasks resume on new Spot instance
       │
       ├──────────────────────────────────────┼──────────────────────────┼──────────────────────────────────────────────────────────────────────────
       ┤
       │ Neon free tier cold start            │ 5s delay after 5min idle │ Acceptable for dev; upgrade to Cloud SQL for production
       │
       ├──────────────────────────────────────┼──────────────────────────┼──────────────────────────────────────────────────────────────────────────
       ┤
       │ ChromaDB data loss on VM             │ Embeddings lost          │ Mount GCS FUSE or use persistent disk with snapshots
       │
       ├──────────────────────────────────────┼──────────────────────────┼──────────────────────────────────────────────────────────────────────────
       ┤
       │ Cloud Run cold start for Spring Boot │ 10-20s first request     │ Set min-instances: 1 (costs ~$15/month)
       │
       ├──────────────────────────────────────┼──────────────────────────┼──────────────────────────────────────────────────────────────────────────
       ┤
       │ Python env conflicts (Phase 3)       │ Model loading failures   │ Separate Docker containers per model group
       │
       ├──────────────────────────────────────┼──────────────────────────┼──────────────────────────────────────────────────────────────────────────
       ┤
       │ GCP credit expiry                    │ Service downtime         │ Monitor credit usage; set billing alerts at 50%, 75%, 90%
       │
       └──────────────────────────────────────┴──────────────────────────┴──────────────────────────────────────────────────────────────────────────
       ┘









       ---
       Critical Files for Implementation

       - D:\kride-project\subproject\SDUI\.github\workflows\deploy.yml -- Current CI/CD pipeline to fork and adapt for GCP Cloud Run deployment
       - D:\kride-project\subproject\SDUI\SDUI-server\Dockerfile -- Existing multi-stage Docker build for Spring Boot, reusable for Cloud Run with
       no changes
       - D:\kride-project\subproject\SDUI\metadata-project\next.config.ts -- Must update BACKEND_URL and /kride-api/* proxy destinations to point to
        GCP endpoints instead of EC2
       - D:\kride-project\src\api\fastapi_server.py -- Core FastAPI server that loads models at startup; needs refactoring to call TorchServe HTTP
       endpoints instead of inline pickle/torch loading (Phase 2)
       - D:\kride-project\subproject\NLP\chatbot\config.py -- ChromaDB path configuration must change from local filesystem to ChromaDB server mode
       (HTTP client) for Phase 2+ when services are split across Cloud Run and GPU VM
