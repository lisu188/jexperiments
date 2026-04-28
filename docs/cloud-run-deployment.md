# Cloud Run Deployment

This repository deploys the Spring Boot `blogsite` module to Cloud Run as a container image. The checked-in `Dockerfile` builds `:blogsite:bootJar`, copies `ExperimentBlogSite-*.jar` into a Java 17 runtime image, and starts Spring Boot on `${PORT:-8080}` so the same image works locally and on Cloud Run.

The deployment path has two modes:

- Local/manual: use `gcloud` to build the image with Cloud Build and deploy it.
- GitHub Actions: use `gh` to configure repository variables/secrets and trigger `.github/workflows/deploy-cloud-run.yml`, which runs the same `gcloud builds submit` and `gcloud run deploy` commands.

## Prerequisites

Install and authenticate both CLIs:

```bash
gcloud auth login
gcloud auth application-default login
gh auth login
```

Choose deployment values:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export REPOSITORY="jexperiments"
export SERVICE="jexperiments-blogsite"
export POOL_ID="github-actions"
export PROVIDER_ID="github"
export SERVICE_ACCOUNT_ID="jexperiments-cloud-run-deployer"
export REPO="lisu188/jexperiments"
```

Enable the Google Cloud APIs used by the deployment:

```bash
gcloud config set project "$PROJECT_ID"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com
```

Create the Artifact Registry repository once:

```bash
gcloud artifacts repositories create "$REPOSITORY" \
  --repository-format=docker \
  --location="$REGION" \
  --description="jexperiments Cloud Run images"
```

## Local `gcloud` Deployment

Build and push a container image with Cloud Build:

```bash
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/blogsite:$(git rev-parse --short HEAD)"
gcloud builds submit --project "$PROJECT_ID" --tag "$IMAGE" .
```

Deploy that image to Cloud Run:

```bash
gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --platform managed \
  --image "$IMAGE" \
  --port 8080 \
  --allow-unauthenticated
```

Print the deployed URL:

```bash
gcloud run services describe "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --platform managed \
  --format='value(status.url)'
```

Use `--no-allow-unauthenticated` instead of `--allow-unauthenticated` if the service should stay private.

## GitHub Actions Setup With `gcloud` and `gh`

The workflow authenticates to Google Cloud through Workload Identity Federation rather than a long-lived service account key.

Create the deployment service account:

```bash
gcloud iam service-accounts create "$SERVICE_ACCOUNT_ID" \
  --project "$PROJECT_ID" \
  --display-name="jexperiments Cloud Run deployer"

export SERVICE_ACCOUNT="${SERVICE_ACCOUNT_ID}@${PROJECT_ID}.iam.gserviceaccount.com"
```

Grant deployment permissions to that service account:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/cloudbuild.builds.editor"

gcloud artifacts repositories add-iam-policy-binding "$REPOSITORY" \
  --project "$PROJECT_ID" \
  --location="$REGION" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/artifactregistry.writer"
```

Create the Workload Identity pool and GitHub OIDC provider:

```bash
export PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"

gcloud iam workload-identity-pools create "$POOL_ID" \
  --project "$PROJECT_ID" \
  --location="global" \
  --display-name="GitHub Actions"

gcloud iam workload-identity-pools providers create-oidc "$PROVIDER_ID" \
  --project "$PROJECT_ID" \
  --location="global" \
  --workload-identity-pool="$POOL_ID" \
  --display-name="GitHub ${REPO}" \
  --issuer-uri="https://token.actions.githubusercontent.com/" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
  --attribute-condition="assertion.repository=='${REPO}'"
```

Allow this repository to impersonate the deployment service account:

```bash
gcloud iam service-accounts add-iam-policy-binding "$SERVICE_ACCOUNT" \
  --project "$PROJECT_ID" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/attribute.repository/${REPO}"
```

Configure the GitHub repository with `gh`:

```bash
export WORKLOAD_IDENTITY_PROVIDER="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/providers/${PROVIDER_ID}"

gh variable set GCP_PROJECT_ID --body "$PROJECT_ID"
gh variable set GCP_REGION --body "$REGION"
gh variable set ARTIFACT_REGISTRY_REPOSITORY --body "$REPOSITORY"
gh variable set CLOUD_RUN_SERVICE --body "$SERVICE"
gh secret set GCP_WORKLOAD_IDENTITY_PROVIDER --body "$WORKLOAD_IDENTITY_PROVIDER"
gh secret set GCP_SERVICE_ACCOUNT --body "$SERVICE_ACCOUNT"
```

After the workflow exists on the default branch, trigger a deployment with `gh`:

```bash
gh workflow run deploy-cloud-run.yml --ref main
gh run list --workflow deploy-cloud-run.yml --limit 1
```

To deploy a custom image tag or keep the service private:

```bash
gh workflow run deploy-cloud-run.yml \
  --ref main \
  -f image_tag="$(git rev-parse --short HEAD)" \
  -f allow_unauthenticated=false
```

## Operational Notes

The workflow builds the Dockerfile from the repository root. The Dockerfile copies the full source tree because `:blogsite:bootJar` generates blog content from every experiment module's `BLOG.md`.

If Cloud Build reports an Artifact Registry permission error, grant `roles/artifactregistry.writer` on the repository to the Cloud Build service account named in the error. Google Cloud projects can differ in which default Cloud Build service account is used.

This deployment path does not change experiment behavior and does not deploy automatically on push. Deployments are manual through `workflow_dispatch`.
