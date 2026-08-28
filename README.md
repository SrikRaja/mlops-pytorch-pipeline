# CIFAR-10 Classifier: Docker + Kubernetes Pipeline

PyTorch image classifier for CIFAR-10, deployed through the full MLOps lifecycle —
local training, containerized inference, and orchestrated serving on Kubernetes.
Built for the MLOps & Infrastructure course assignment.

## Architecture

```
                    ┌─────────────────────┐
                    │   Docker daemon      │
                    │  (docker-outside-    │
                    │   of-docker on the   │
                    │   Codespace host)    │
                    └──────────┬───────────┘
                               │ builds
                    ┌──────────┴───────────┐
                    │                      │
          mlops-train:v1          mlops-serve:v1
          (training image)        (serving image)
                    │                      │
                    │ k3d image import     │
                    ▼                      ▼
        ┌─────────────────────────────────────────┐
        │       k3d cluster (single node)          │
        │                                          │
        │   namespace: ml-training                 │
        │   ┌─────────────┐   ┌──────────────────┐ │
        │   │ Job:        │   │ Deployment:       │ │
        │   │ model-      │   │ model-serving     │ │
        │   │ training    │   │ (2 replicas)      │ │
        │   └──────┬──────┘   └────────┬──────────┘ │
        │          │ writes            │ reads       │
        │          ▼                   ▼             │
        │   ┌──────────────────────────────────┐    │
        │   │ PVC: checkpoints-pvc (1Gi)        │    │
        │   └──────────────────────────────────┘    │
        │                                            │
        │   PVC: training-data-pvc (2Gi) — CIFAR-10  │
        │   ConfigMap: training-config                │
        └─────────────────────┬────────────────────┘
                               │ Service (ClusterIP :80 -> :8080)
                               ▼
                     kubectl port-forward
                               │
                               ▼
                          curl / client
```

The training Job runs once, writes a checkpoint to a shared PVC, and exits. The
serving Deployment mounts that same PVC read-only and loads whatever checkpoint
is there at startup — the two never talk to each other directly, they just share
storage.

## Repo structure

```
mlops-pytorch-pipeline/
├── src/                # model, dataset, train, serve
├── configs/             # training_config.yaml (local/Docker use)
├── docker/               # Dockerfile.train, Dockerfile.serve
├── k8s/                   # namespace, configmap, pvc, job, deployment, service, hpa
├── requirements/           # train.txt, serve.txt (CPU-only torch wheels)
├── tests/
└── .devcontainer/           # docker-outside-of-docker + k3d/kubectl setup
```

## Local setup

This was built and tested entirely in a GitHub Codespace. The devcontainer installs
`docker-outside-of-docker` (talks to the host's real Docker daemon via the mounted
socket) plus `kubectl` and `k3d` for a local Kubernetes cluster.

```bash
k3d cluster create mycluster
```

**Note on `kind` vs `k3d`:** I originally set this up with `kind`, but its
`kubeadm`-based control-plane bootstrap consistently raced and failed on this
particular Codespace VM (the API server came up ~5-10 seconds slower than `kind`'s
internal taint-removal step allows for, with no retry). Five clean attempts, same
failure every time. Switched to `k3d`, which uses k3s instead of full kubeadm —
lighter bootstrap, no separate etcd process, and it's never failed once since.

### Build and run training locally

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1
```

### Build and run serving locally

```bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run --rm -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1

curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Running on Kubernetes

Images have to be imported into the k3d node explicitly — it doesn't see the
host's `docker build` output on its own:

```bash
k3d image import mlops-train:v1 -c mycluster
k3d image import mlops-serve:v1 -c mycluster
```

Then apply everything in order:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/training-job.yaml
kubectl get pods -n ml-training -w   # wait for Completed

kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl get pods -n ml-training -w   # wait for 2/2 Running
```

Test it:

```bash
kubectl port-forward svc/model-serving 8080:80 -n ml-training

# in another terminal
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## A couple of things worth knowing

- **Image size**: the first build used the default PyTorch wheels, which pull in
  full CUDA binaries even though this runs CPU-only — 5.4GB per image. Switched
  `requirements/*.txt` to `--extra-index-url https://download.pytorch.org/whl/cpu`,
  which dropped both images to ~1.1GB. This also fixed a real problem: two 5.4GB
  images didn't both fit in the k3d node's disk at once, so the older one kept
  getting silently evicted between imports.
- **Epochs**: `training_config.yaml` (used for local/Docker runs) trains for 10
  epochs. The in-cluster ConfigMap (`k8s/configmap.yaml`) is set to 2 epochs —
  full CPU-only training isn't really the point of the Kubernetes demonstration,
  and 2 epochs is enough to prove the whole pipeline (Job -> checkpoint -> PVC ->
  Deployment -> prediction) actually works.
- **Checkpoint selection**: training only saves a checkpoint when validation loss
  improves, so on the 2-epoch cluster run, epoch 2 (higher train accuracy, worse
  val loss) didn't overwrite epoch 1's checkpoint. The serving Deployment loads
  whatever the best checkpoint was, not necessarily the most recent one.