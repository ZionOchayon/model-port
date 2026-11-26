# MinIO community server built from source at the last community release (the upstream image is gone; ADR-0003).
ARG GO_BUILDER_IMAGE
ARG ALPINE_IMAGE

FROM ${GO_BUILDER_IMAGE} AS build
ARG MINIO_RELEASE
RUN test -n "${MINIO_RELEASE}" \
    && apk add --no-cache git \
    && CGO_ENABLED=0 GOFLAGS=-trimpath go install -ldflags "-s -w" "github.com/minio/minio@${MINIO_RELEASE}"

FROM ${ALPINE_IMAGE}
RUN apk add --no-cache ca-certificates && adduser -D -u 1000 minio
COPY --from=build /go/bin/minio /usr/bin/minio
USER 1000
EXPOSE 9000
ENTRYPOINT ["/usr/bin/minio"]
CMD ["server", "/data", "--address", ":9000"]
