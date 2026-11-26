# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Repository skeleton: uv workspace with the `modelport` SDK and `modelport-platform` packages,
  pinned `versions.env`, `Makefile`, license and dataset credits.
- Development toolbox: `docker compose run --rm toolbox make <target>` runs every target with the
  pinned tools and only Docker on the host (ADR-0019).
