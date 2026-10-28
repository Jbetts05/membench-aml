# Infrastructure setup

Use `setup.ps1` to create the Azure runtime environment for the benchmark notebook.

The script creates or configures:

- Azure resource group
- Azure AI Services account
- Azure AI Foundry project
- Azure ML workspace
- Azure ML compute with a system-assigned managed identity
- Storage settings required by AML command jobs
- RBAC for the current user and AML compute identity
- `.env` with the generated Azure resource values

The notebook registers the reusable AML job environment from `infra\memorybench-runner-conda.yml`. That environment installs the runner dependencies once as an Azure ML environment asset instead of installing packages inside every job.

The script does **not** deploy models. After it finishes, deploy or select your answer, judge, and embedding deployments in the generated AI Services resource, then fill these values in `.env`:

```text
ANSWER_DEPLOYMENT=
JUDGE_DEPLOYMENT=
EMBEDDING_DEPLOYMENT=
```

## Run

Prerequisite: Azure CLI 2.83.0 or newer.

```powershell
az login
az upgrade
.\infra\setup.ps1 -EnvironmentName memorybench -Location eastus2
```

The script writes `.env` at the repo root. Existing model deployment values are preserved when `.env` already exists.
Use `-Force` to regenerate `.env` without preserving existing model deployment values.

## Parameters

Common overrides:

```powershell
.\infra\setup.ps1 `
  -EnvironmentName memorybench `
  -Location eastus2 `
  -MlComputeName memorybench-cpu `
  -ComputeSize Standard_DS3_v2 `
  -ComputeMaxNodes 2
```

Power users can also pass explicit resource names:

```powershell
.\infra\setup.ps1 `
  -ResourceGroupName rg-memorybench `
  -MlWorkspaceName memorybench-aml `
  -AiServicesName memorybench-ai-12345 `
  -AiProjectName memorybench-project
```
