# Lightseed Studio — Railway edge container.
#
# Build context and root directory: deploy/railway
#   podman build -t lightseed-railway deploy/railway
#
# Contents: Tailscale (userspace, no TUN, no privileged mode), oauth2-proxy
# (Google sign-in), and a small Node gateway that is the only thing allowed to
# talk to the PC. Both binaries are pinned by version AND sha256: the image is a
# public trust boundary, so a silently swapped tarball must break the build.

ARG NODE_IMAGE=node:22-bookworm-slim@sha256:de170b39439eb0e5a3444ad7d60b015614a381e10fecae2e30ec9e735122f38d

# ---------------------------------------------------------------- fetch ---

FROM ${NODE_IMAGE} AS fetch

ARG OAUTH2_PROXY_VERSION=v7.15.5
ARG OAUTH2_PROXY_SHA256=f63f94bf72c5f46ab002a0a275aa8b3cf19b4d828aed08a13978cb9a62c3a1fd
ARG TAILSCALE_VERSION=1.104.2
ARG TAILSCALE_SHA256=512f613ed7d04fc7282f7dc5bd76aa105aae28168fbb9357eb749ebcbc29f0ae

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    mkdir -p /out; \
    curl --proto '=https' --tlsv1.2 -fsSL --retry 5 --retry-delay 2 \
      -o /tmp/oauth2-proxy.tar.gz \
      "https://github.com/oauth2-proxy/oauth2-proxy/releases/download/${OAUTH2_PROXY_VERSION}/oauth2-proxy-${OAUTH2_PROXY_VERSION}.linux-amd64.tar.gz"; \
    echo "${OAUTH2_PROXY_SHA256}  /tmp/oauth2-proxy.tar.gz" | sha256sum -c -; \
    tar -xzf /tmp/oauth2-proxy.tar.gz -C /tmp; \
    install -m 0755 "/tmp/oauth2-proxy-${OAUTH2_PROXY_VERSION}.linux-amd64/oauth2-proxy" /out/oauth2-proxy

RUN set -eux; \
    curl --proto '=https' --tlsv1.2 -fsSL --retry 5 --retry-delay 2 \
      -o /tmp/tailscale.tgz \
      "https://pkgs.tailscale.com/stable/tailscale_${TAILSCALE_VERSION}_amd64.tgz"; \
    echo "${TAILSCALE_SHA256}  /tmp/tailscale.tgz" | sha256sum -c -; \
    tar -xzf /tmp/tailscale.tgz -C /tmp; \
    install -m 0755 "/tmp/tailscale_${TAILSCALE_VERSION}_amd64/tailscaled" /out/tailscaled; \
    install -m 0755 "/tmp/tailscale_${TAILSCALE_VERSION}_amd64/tailscale" /out/tailscale

# --------------------------------------------------------------- runtime ---

FROM ${NODE_IMAGE}

ARG LIGHTSEED_VERSION=unknown

LABEL org.opencontainers.image.title="lightseed-railway" \
      org.opencontainers.image.description="Lightseed Studio edge: Google sign-in, Tailscale, reverse proxy to the owner's PC" \
      org.opencontainers.image.version="${LIGHTSEED_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive

# ca-certificates: oauth2-proxy talks TLS to Google, tailscaled to the control
# plane. Neither the slim base nor a distro-less image ships them.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    update-ca-certificates

COPY --from=fetch /out/oauth2-proxy /out/tailscaled /out/tailscale /usr/local/bin/

# Unprivileged uid with a percent-free home: tailscaled writes its netcheck
# state under $HOME, and the image is a public box -- it gets no root.
RUN set -eux; \
    groupadd --gid 10001 lightseed; \
    useradd --uid 10001 --gid 10001 --home-dir /var/lib/lightseed \
      --create-home --shell /usr/sbin/nologin lightseed

WORKDIR /app
COPY gateway.js entrypoint.sh oauth2-proxy.args /app/
COPY pages /app/pages
COPY templates /app/templates
COPY assets /app/assets
RUN set -eux; \
    chmod 0755 /app/entrypoint.sh; \
    chmod 0644 /app/gateway.js /app/oauth2-proxy.args; \
    mkdir -p /var/lib/lightseed /tmp/lightseed-ts; \
    chown -R lightseed:lightseed /app /var/lib/lightseed /tmp/lightseed-ts

ENV HOME=/var/lib/lightseed \
    NODE_ENV=production \
    LIGHTSEED_VERSION=${LIGHTSEED_VERSION} \
    LIGHTSEED_GATEWAY_PORT=18080 \
    LIGHTSEED_TAILSCALE_PROXY=127.0.0.1:1055 \
    LIGHTSEED_TAILSCALED_SOCKET=/tmp/lightseed-ts/ts.sock \
    LIGHTSEED_HOSTNAME=lightseed-railway

USER lightseed

# Railway routes its public domain to $PORT. Nothing listens there but
# oauth2-proxy; the gateway is loopback-only so no client can skip sign-in.
EXPOSE 8080

# Railway decides liveness from GET /healthz via railway.toml; a Docker
# HEALTHCHECK here would only add a second, disagreeing opinion.
CMD ["/app/entrypoint.sh"]
