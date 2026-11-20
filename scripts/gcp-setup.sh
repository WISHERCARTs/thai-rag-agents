#!/usr/bin/env bash
# One-time Google Cloud setup for the thai-rag-agents live demo. Run it in Cloud Shell, in the same
# project as thai-llm-gateway, after the gateway's own scripts/gcp-setup.sh:
#
#   gcloud config set project gen-lang-client-0786555487
#   curl -fsSL https://raw.githubusercontent.com/WISHERCARTs/thai-rag-agents/main/scripts/gcp-setup.sh -o rag-setup.sh
#   bash rag-setup.sh
#
# It asks for nothing and prints no secret. It is safe to run again: every step checks what exists.
set -euo pipefail

REGION="${REGION:-asia-southeast1}"
GITHUB_REPO="${GITHUB_REPO:-WISHERCARTs/thai-rag-agents}"
AR_REPO="rag-agents"
RUNTIME_SA_NAME="rag-agents-runtime"
DEPLOY_SA_NAME="rag-agents-deployer"
POOL="llm-gateway-github"  # the gateway's pool; this repo gets its own provider in it
PROVIDER="rag-agents"
KEY_SECRET="RAG_AGENTS_GATEWAY_KEY"

PROJECT_ID="$(gcloud config get-value project 2>/dev/null)"
if [[ -z "$PROJECT_ID" ]]; then
  echo "Set the project first: gcloud config set project YOUR_PROJECT_ID" >&2
  exit 1
fi
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
RUNTIME_SA="${RUNTIME_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOY_SA="${DEPLOY_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

step() { printf '\n== %s\n' "$*"; }
# A new service account takes a short while to be visible to IAM, so bindings are retried.
retry() {
  local n
  for n in 1 2 3 4 5 6; do
    "$@" && return 0
    echo "  not ready yet, retrying in $((n * 5))s" >&2
    sleep $((n * 5))
  done
  "$@"
}

step "Project ${PROJECT_ID} (${PROJECT_NUMBER}), region ${REGION}"
if ! gcloud secrets describe LLM_GATEWAY_GATEWAY_CLIENT_KEYS >/dev/null 2>&1; then
  echo "No LLM_GATEWAY_GATEWAY_CLIENT_KEYS secret here. Run thai-llm-gateway's scripts/gcp-setup.sh first," >&2
  echo "in the same project." >&2
  exit 1
fi

step "Artifact Registry repository for the image"
gcloud artifacts repositories describe "$AR_REPO" --location "$REGION" >/dev/null 2>&1 ||
  gcloud artifacts repositories create "$AR_REPO" --location "$REGION" --repository-format docker \
    --description "thai-rag-agents images" --quiet

step "Service accounts"
for sa in "$RUNTIME_SA_NAME:the Cloud Run service" "$DEPLOY_SA_NAME:GitHub Actions deploys"; do
  name="${sa%%:*}"
  gcloud iam service-accounts describe "${name}@${PROJECT_ID}.iam.gserviceaccount.com" >/dev/null 2>&1 ||
    gcloud iam service-accounts create "$name" --display-name "thai-rag-agents: ${sa#*:}" --quiet
done
# run.admin lets the deploy make the service public (--allow-unauthenticated).
retry gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:${DEPLOY_SA}" \
  --role roles/run.admin --condition None --quiet >/dev/null
# The deployer may push to this repository only, and run the service as the runtime account only.
retry gcloud artifacts repositories add-iam-policy-binding "$AR_REPO" --location "$REGION" \
  --member "serviceAccount:${DEPLOY_SA}" --role roles/artifactregistry.writer --quiet >/dev/null
retry gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" --member "serviceAccount:${DEPLOY_SA}" \
  --role roles/iam.serviceAccountUser --quiet >/dev/null

step "Workload Identity Federation: GitHub Actions in ${GITHUB_REPO} only, no JSON keys"
gcloud iam workload-identity-pools providers describe "$PROVIDER" --location global \
  --workload-identity-pool "$POOL" >/dev/null 2>&1 ||
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --location global \
    --workload-identity-pool "$POOL" --display-name "GitHub thai-rag-agents" \
    --issuer-uri "https://token.actions.githubusercontent.com" \
    --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
    --attribute-condition "assertion.repository == '${GITHUB_REPO}'" --quiet
POOL_ID="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}"
retry gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_SA" --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${GITHUB_REPO}" --quiet >/dev/null

step "Secrets"
grant() { # secret, service account
  retry gcloud secrets add-iam-policy-binding "$1" --member "serviceAccount:$2" \
    --role roles/secretmanager.secretAccessor --quiet >/dev/null
}
# The gateway generated a client key for this app; copy it into its own secret without showing it.
if ! gcloud secrets describe "$KEY_SECRET" >/dev/null 2>&1; then
  gcloud secrets create "$KEY_SECRET" --replication-policy automatic --quiet >/dev/null
fi
if [[ -z "$(gcloud secrets versions list "$KEY_SECRET" --filter state=enabled --format 'value(name)' --limit 1)" ]] ||
  [[ "${RESET:-}" == *"$KEY_SECRET"* ]]; then
  key="$(gcloud secrets versions access latest --secret LLM_GATEWAY_GATEWAY_CLIENT_KEYS |
    tr ',' '\n' | sed -n 's/^thai-rag-agents://p' | tr -d '\n')"
  if [[ -z "$key" ]]; then
    echo "  LLM_GATEWAY_GATEWAY_CLIENT_KEYS has no thai-rag-agents key; add one there first." >&2
    exit 1
  fi
  printf '%s' "$key" | gcloud secrets versions add "$KEY_SECRET" --data-file - --quiet >/dev/null
  unset key
  echo "  saved ${KEY_SECRET} (the gateway's thai-rag-agents client key)"
else
  echo "  ${KEY_SECRET} already set (run with RESET=${KEY_SECRET} to copy it again)"
fi
# The runtime reads the key, the demo's rate-limit counters (the gateway's Upstash) and the IP hash salt.
for secret in "$KEY_SECRET" LLM_GATEWAY_UPSTASH_REDIS_REST_URL LLM_GATEWAY_UPSTASH_REDIS_REST_TOKEN LLM_GATEWAY_USER_HASH_SALT; do
  grant "$secret" "$RUNTIME_SA"
done
# The deployer uses the key to build the search index through the gateway, and nothing else.
grant "$KEY_SECRET" "$DEPLOY_SA"
echo "  access granted"

step "Done. deploy.yml already has these values; tell Claude if any differs:"
echo "GCP_PROJECT_ID=${PROJECT_ID}"
echo "GCP_WIF_PROVIDER=${POOL_ID}/providers/${PROVIDER}"
echo "GCP_DEPLOY_SA=${DEPLOY_SA}"
