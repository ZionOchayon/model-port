# ADR-0019: A Docker Compose toolbox for development tools; the platform stays on kind

- Status: Accepted
- Date: 2026-11-26

## Context

Every `make` target needs the same pinned tools: uv, Python 3.12, kind, kubectl, Helm, the Docker CLI, git, gh, and `promtool`/`amtool` for rule and Alertmanager checks. Installing them on each developer machine is slow, differs by operating system (Windows and macOS have no `make`, and tool builds differ), and drifts from the versions in `versions.env`.

The platform itself (Airflow, Ray, MLflow, MinIO, PostgreSQL, Prometheus) runs on a kind cluster, which already runs inside Docker. Running the platform with Compose instead would mean a second deployment path that does not test the Kubernetes manifests, RayJobs or the RayService.

## Decision

- Add one Compose service, `toolbox`, built from `deploy/images/toolbox/Dockerfile`: Ubuntu 24.04 (pinned by digest) with the pinned uv and Python 3.12, kind, kubectl and Helm (sha256-checked downloads, the same check as `scripts/install-tools.sh`), the Docker CLI, git, gh, and `promtool`/`amtool` copied from the pinned Prometheus and Alertmanager images.
- The service mounts the repository at `/workspace` and the host's Docker socket, joins the external Docker network `kind`, publishes the UI ports (8080, 5000, 8265, 8000, 3000, 9090, 9093, 9000) and keeps the uv cache in a named volume. Its virtual environment lives outside the bind mount, so it never clashes with a host `.venv`.
- Inside the toolbox, kubectl talks to the cluster through `kind get kubeconfig --internal --name modelport` (the control-plane container's name on the `kind` network), and `scripts/ui.sh` port-forwards with `--address 0.0.0.0` so the host browser reaches the UIs through the published ports.
- Every `make` target works in both places: `docker compose run --rm toolbox make <target>` on any machine that has Docker, and `make <target>` on a Linux host with the tools installed.
- The platform stays on kind. CI keeps running the tools directly on the runner, with no toolbox, so CI tests exactly the documented host path.

## Consequences

- A new developer needs only Docker. The first run builds the toolbox image once.
- Tool versions come from `versions.env` in the image build, so the toolbox and CI use the same pins, and `scripts/check_versions.py` checks both.
- The toolbox controls the host's Docker engine through the socket. That is the same access kind needs, and it is a development environment.
- Paths that `docker run -v` mounts from inside the toolbox would point at the toolbox's file system, not the host's. Targets therefore run `promtool` and `amtool` directly when they are on the `PATH` (always inside the toolbox) and fall back to `docker run` on a host.
- One extra image to keep pinned. The weekly CI run does not build it; a broken toolbox shows up on the next local build.
