#!/usr/bin/env bash
set -euo pipefail

curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash

k3d version
kubectl version --client
