#!/usr/bin/env node
"use strict";

// Lightseed Studio: Railway edge gateway.
//
// Two modes:
//
//   proxy (production)  Listen on 127.0.0.1:$LIGHTSEED_GATEWAY_PORT; the only
//                       peer is oauth2-proxy on the same loopback. Identity
//                       comes from the X-Forwarded-Email / X-Forwarded-User
//                       headers oauth2-proxy sets -- that is, from the Google
//                       id_token, never from the client.
//   dev                 Listen on 0.0.0.0:$PORT with no sign-in layer at all;
//                       identity comes from LIGHTSEED_DEV_EMAIL/-SUB. Only for
//                       local smoke runs.
//
// In both modes the request is forwarded to the owner's PC over the Tailscale
// userspace HTTP proxy. Client-sent X-Lightseed-* headers are stripped and the
// three the PC trusts are injected. When the PC cannot be reached the reply is
// the branded offline page with 503 -- never a raw proxy error.
//
// Zero dependencies: node:http, node:crypto, node:fs, node:path only.

const http = require("node:http");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

const WEB_ERROR_SCHEMA = "qwen_engine.web_error.v1";
const HEALTH_SCHEMA = "lightseed.gateway_health.v1";

// ---------------------------------------------------------------- config ---

function envText(name) {
  const raw = process.env[name];
  return raw === undefined ? "" : raw.trim();
}

function envInt(name, fallback, minimum) {
  const raw = envText(name);
  if (raw === "") {
    return fallback;
  }
  if (!/^\d{1,9}$/.test(raw)) {
    throw new Error(`${name} must be a plain non-negative integer`);
  }
  const value = Number(raw);
  if (value < minimum) {
    throw new Error(`${name} must be at least ${minimum}`);
  }
  return value;
}

function isTruthy(value) {
  const lowered = value.toLowerCase();
  return lowered === "1" || lowered === "true" || lowered === "yes";
}

function parseHostPort(value, what) {
  const text = value.trim();
  if (text === "" || text.includes("/") || text.includes("@") ||
      text.includes("?") || text.includes("#")) {
    throw new Error(`${what} must be host:port, without a scheme or path`);
  }
  let host;
  let portText;
  if (text.startsWith("[")) {
    const end = text.indexOf("]");
    if (end < 0 || text[end + 1] !== ":") {
      throw new Error(`${what} must be [ipv6]:port or host:port`);
    }
    host = text.slice(1, end);
    portText = text.slice(end + 2);
  } else {
    const colon = text.lastIndexOf(":");
    if (colon <= 0) {
      throw new Error(`${what} must be host:port`);
    }
    host = text.slice(0, colon);
    portText = text.slice(colon + 1);
  }
  if (!/^[A-Za-z0-9._:-]+$/.test(host)) {
    throw new Error(`${what} host is not a plain address`);
  }
  if (!/^\d{1,5}$/.test(portText)) {
    throw new Error(`${what} port is not a decimal number`);
  }
  const port = Number(portText);
  if (port < 1 || port > 65535) {
    throw new Error(`${what} port is out of range`);
  }
  return { host, port };
}

function loadConfig() {
  const modeText = envText("LIGHTSEED_MODE").toLowerCase();
  const devMode = modeText === "dev" || isTruthy(envText("LIGHTSEED_DEV_MODE"));
  if (modeText !== "" && modeText !== "dev" && modeText !== "proxy") {
    throw new Error("LIGHTSEED_MODE must be 'dev' or 'proxy'");
  }
  const mode = devMode ? "dev" : "proxy";

  const upstreamText = envText("LIGHTSEED_UPSTREAM");
  if (upstreamText === "") {
    throw new Error("LIGHTSEED_UPSTREAM is required (host:port of the PC)");
  }
  const upstream = parseHostPort(upstreamText, "LIGHTSEED_UPSTREAM");

  const proxyToken = envText("LIGHTSEED_PROXY_TOKEN");
  if (proxyToken === "") {
    throw new Error("LIGHTSEED_PROXY_TOKEN is required");
  }

  const port = envInt("PORT", 0, 1);
  if (mode === "dev" && port === 0) {
    throw new Error("PORT is required in dev mode");
  }

  // Two fields, never a "host:port" string: node treats a string as a UNIX
  // socket path and would silently create a file called "0.0.0.0:8080".
  const listenHost = mode === "dev" ? "0.0.0.0" : "127.0.0.1";
  const listenPort = mode === "dev"
    ? port
    : envInt("LIGHTSEED_GATEWAY_PORT", 18080, 1);

  const noTailscale = isTruthy(envText("LIGHTSEED_NO_TAILSCALE"));

  const devEmail = envText("LIGHTSEED_DEV_EMAIL").toLowerCase();
  const devSub = envText("LIGHTSEED_DEV_SUB");
  if (mode === "dev") {
    if (!isEmailish(devEmail)) {
      throw new Error("LIGHTSEED_DEV_EMAIL is required in dev mode");
    }
  }

  return {
    mode,
    listenHost,
    listenPort,
    upstream,
    upstreamText,
    proxyToken,
    useTailscaleProxy: !noTailscale,
    tailscaleProxy: parseHostPort(
      envText("LIGHTSEED_TAILSCALE_PROXY") || "127.0.0.1:1055",
      "LIGHTSEED_TAILSCALE_PROXY"),
    tailscaledSocket: envText("LIGHTSEED_TAILSCALED_SOCKET") ||
      "/tmp/lightseed-ts/ts.sock",
    devEmail,
    devSub,
    version: envText("LIGHTSEED_VERSION") || "unknown",
    generalLimit: envInt("LIGHTSEED_RATE_LIMIT_MAX", 30, 1),
    generalWindowMs: 10_000,
    jobLimit: envInt("LIGHTSEED_JOB_RATE_LIMIT_MAX", 5, 1),
    jobWindowMs: 60_000,
    connectTimeoutMs: envInt("LIGHTSEED_CONNECT_TIMEOUT_MS", 10_000, 1),
    idleTimeoutMs: envInt("LIGHTSEED_UPSTREAM_IDLE_TIMEOUT_MS", 330_000, 1),
    healthTimeoutMs: envInt("LIGHTSEED_HEALTH_TIMEOUT_MS", 3_000, 1),
    healthCacheMs: envInt("LIGHTSEED_HEALTH_CACHE_MS", 5_000, 0),
    offlineSniffBytes: envInt("LIGHTSEED_OFFLINE_SNIFF_BYTES", 16 * 1024, 1),
    logRequests: isTruthy(envText("LIGHTSEED_LOG_REQUESTS")),
  };
}

function isEmailish(value) {
  if (value.length === 0 || value.length > 320) {
    return false;
  }
  // Bounded, printable ASCII only: the PC lowercases and stores this string,
  // so no control characters, spaces or header-splitting material.
  if (!/^[\x21-\x7e]+$/.test(value)) {
    return false;
  }
  const at = value.indexOf("@");
  return at > 0 && at === value.lastIndexOf("@") &&
    value.indexOf(".", at) > at + 1 && !value.endsWith(".");
}

function isSafeId(value) {
  return value.length > 0 && value.length <= 255 &&
    /^[\x21-\x7e]+$/.test(value);
}

function isLoopback(address) {
  return address === "127.0.0.1" || address === "::1" ||
    address === "::ffff:127.0.0.1";
}

// ------------------------------------------------------------------ pages ---

function readText(relative) {
  return fs.readFileSync(path.join(__dirname, relative), "utf8");
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function buildOfflinePage() {
  const template = readText("pages/offline.html");
  const logo = readText("assets/lightseed-logo-on-dark.svg");
  // Function replacements: the logo must never be read as a replacement
  // pattern ($& et al).
  return template
    .replace(/\{\{LOGO\}\}/g, () => logo)
    .replace(/\{\{VERSION\}\}/g, () => escapeHtml(config.version));
}

// ------------------------------------------------------------------ state ---

const config = loadConfig();
const offlinePage = Buffer.from(buildOfflinePage(), "utf8");

function log(message, fields) {
  let line = `${new Date().toISOString()} gateway ${message}`;
  if (fields) {
    for (const [key, value] of Object.entries(fields)) {
      line += ` ${key}=${JSON.stringify(value)}`;
    }
  }
  process.stderr.write(`${line}\n`);
}

// Logs identify a person by a short hash, never by address: Railway logs are
// not a place for other people's email addresses.
function userTag(email) {
  if (!email) {
    return "-";
  }
  return crypto.createHash("sha256").update(email).digest("hex").slice(0, 12);
}

function jsonReply(res, status, payload, extraHeaders) {
  const body = Buffer.from(JSON.stringify(payload), "utf8");
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": String(body.length),
    "cache-control": "no-store",
    ...(extraHeaders || {}),
  });
  res.end(body);
}

function webError(res, status, code, message, extraHeaders) {
  jsonReply(res, status, { schema: WEB_ERROR_SCHEMA, code, message },
    extraHeaders);
}

function sendOffline(res) {
  if (res.headersSent || res.writableEnded) {
    res.destroy();
    return;
  }
  res.writeHead(503, {
    "content-type": "text/html; charset=utf-8",
    "content-length": String(offlinePage.length),
    "cache-control": "no-store",
    "retry-after": "60",
    "x-lightseed-offline": "1",
  });
  res.end(offlinePage);
}

// -------------------------------------------------------------- rate limit ---

class WindowLimiter {
  constructor(max, windowMs) {
    this.max = max;
    this.windowMs = windowMs;
    this.hits = new Map();
    this.sweepAt = Date.now() + windowMs;
  }

  // Returns 0 when the request may proceed, else the seconds to wait.
  take(key) {
    const now = Date.now();
    if (now >= this.sweepAt) {
      for (const [existing, times] of this.hits) {
        const live = times.filter((t) => now - t < this.windowMs);
        if (live.length === 0) {
          this.hits.delete(existing);
        } else {
          times.length = 0;
          times.push(...live);
        }
      }
      this.sweepAt = now + this.windowMs;
    }
    const times = this.hits.get(key) || [];
    while (times.length > 0 && now - times[0] >= this.windowMs) {
      times.shift();
    }
    if (times.length >= this.max) {
      this.hits.set(key, times);
      return Math.max(1, Math.ceil((this.windowMs - (now - times[0])) / 1000));
    }
    times.push(now);
    this.hits.set(key, times);
    return 0;
  }
}

const generalLimiter = new WindowLimiter(config.generalLimit,
  config.generalWindowMs);
const jobLimiter = new WindowLimiter(config.jobLimit, config.jobWindowMs);

function isJobSubmission(method, target) {
  return method === "POST" && target.split("?")[0] === "/api/v1/jobs";
}

// ----------------------------------------------------------------- health ---

const UPSTREAM_PROBE_PATH = "/api/v1/status";

function requestOptions(target, headers, transport) {
  const options = {
    method: "GET",
    path: target,
    headers,
  };
  if (transport === "proxy") {
    options.host = config.tailscaleProxy.host;
    options.port = config.tailscaleProxy.port;
    options.path = `http://${config.upstreamText}${target}`;
  } else {
    options.host = config.upstream.host;
    options.port = config.upstream.port;
  }
  options.headers = { ...headers, host: config.upstreamText };
  return options;
}

function tailscaleState() {
  if (!config.useTailscaleProxy) {
    return Promise.resolve("disabled");
  }
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
      if (!settled) {
        settled = true;
        resolve(value);
      }
    };
    const request = http.request({
      socketPath: config.tailscaledSocket,
      path: "/localapi/v0/status",
      method: "GET",
      // The localapi listener routes on this Host header; without it the
      // reply is a 403.
      headers: { host: "local-tailscaled.sock" },
      timeout: config.healthTimeoutMs,
    }, (response) => {
      const chunks = [];
      let size = 0;
      response.on("data", (chunk) => {
        size += chunk.length;
        if (size <= 1 << 20) {
          chunks.push(chunk);
        }
      });
      response.on("end", () => {
        if (response.statusCode !== 200) {
          finish("unknown");
          return;
        }
        try {
          const status = JSON.parse(Buffer.concat(chunks).toString("utf8"));
          finish(typeof status.BackendState === "string"
            ? status.BackendState
            : "unknown");
        } catch {
          finish("unknown");
        }
      });
      response.on("error", () => finish("unknown"));
    });
    request.on("timeout", () => {
      request.destroy();
      finish("unknown");
    });
    request.on("error", () => finish("unknown"));
    request.end();
  });
}

function upstreamReachable() {
  const transport = config.useTailscaleProxy ? "proxy" : "direct";
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
      if (!settled) {
        settled = true;
        resolve(value);
      }
    };
    const request = http.request(
      requestOptions(UPSTREAM_PROBE_PATH, {}, transport),
      (response) => {
        // HEAD is not used: some upstreams answer it with a body-less 405 but
        // the cheap proof we want is simply "a response came back".
        response.resume();
        // The Tailscale proxy answers 502 when the PC itself is unreachable.
        finish(response.statusCode !== 502 && response.statusCode !== 504);
      });
    request.setTimeout(config.healthTimeoutMs, () => {
      request.destroy();
      finish(false);
    });
    request.on("error", () => finish(false));
    request.end();
  });
}

let healthCache = { at: 0, value: null, inflight: null };

function healthSnapshot() {
  const now = Date.now();
  if (healthCache.value && now - healthCache.at < config.healthCacheMs) {
    return Promise.resolve(healthCache.value);
  }
  if (healthCache.inflight) {
    return healthCache.inflight;
  }
  const inflight = (async () => {
    const [tailscale, reachable] = await Promise.all([
      tailscaleState(),
      upstreamReachable(),
    ]);
    const snapshot = {
      tailscale,
      reachable,
      at: Date.now(),
    };
    healthCache = { at: snapshot.at, value: snapshot, inflight: null };
    return snapshot;
  })().catch(() => {
    healthCache.inflight = null;
    return { tailscale: "unknown", reachable: false, at: Date.now() };
  });
  healthCache.inflight = inflight;
  return inflight;
}

async function handleHealth(req, res) {
  const snapshot = await healthSnapshot();
  const tailscaleUp = snapshot.tailscale === "Running" ||
    snapshot.tailscale === "disabled";
  let status;
  let httpStatus;
  if (!tailscaleUp) {
    status = "tailscale_down";
    httpStatus = 503;
  } else if (snapshot.reachable) {
    status = "ok";
    httpStatus = 200;
  } else {
    // The PC being switched off is a normal state, not a broken container:
    // answer 200 so Railway does not restart the service in a loop.
    status = "degraded";
    httpStatus = 200;
  }
  const payload = {
    schema: HEALTH_SCHEMA,
    status,
    tailscale: snapshot.tailscale === "Running" ? "running" : snapshot.tailscale,
    upstream: snapshot.reachable ? "reachable" : "unreachable",
    version: config.version,
  };
  const body = Buffer.from(JSON.stringify(payload), "utf8");
  res.writeHead(httpStatus, {
    "content-type": "application/json; charset=utf-8",
    "content-length": String(body.length),
    "cache-control": "no-store",
  });
  res.end(req.method === "HEAD" ? undefined : body);
}

// --------------------------------------------------------------- identity ---

function identityFor(req) {
  if (config.mode === "dev") {
    return { email: config.devEmail, sub: config.devSub };
  }
  const remote = req.socket.remoteAddress || "";
  if (!isLoopback(remote)) {
    // The listener is loopback-only; this branch means something else in the
    // container reached it. No identity, no service.
    return { error: "auth_required" };
  }
  const email = (req.headers["x-forwarded-email"] || "").trim().toLowerCase();
  const sub = (req.headers["x-forwarded-user"] || "").trim();
  if (!isEmailish(email)) {
    return { error: "auth_required" };
  }
  return { email, sub: isSafeId(sub) ? sub : "" };
}

// ---------------------------------------------------------------- headers ---

const HOP_BY_HOP = new Set([
  "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
  "te", "trailer", "transfer-encoding", "upgrade",
]);

// Headers that carry the proxy's own trust decisions; forwarding them would
// invite the PC to trust a second, unvetted copy. The gateway re-injects its
// own X-Lightseed-* set instead.
const DROPPED_REQUEST_HEADERS = new Set([
  "host", "x-forwarded-user", "x-forwarded-email", "x-forwarded-preferred-user",
  "x-forwarded-access-token", "x-auth-request-user", "x-auth-request-email",
  "x-auth-request-access-token", "gap-auth", "gap-signature",
]);

const SESSION_COOKIE_NAME = "_lightseed_session";

function stripSessionCookie(value) {
  return value
    .split(";")
    .map((part) => part.trim())
    .filter((part) => part !== "")
    .filter((part) => part.split("=")[0].trim() !== SESSION_COOKIE_NAME)
    .join("; ");
}

function forwardRequestHeaders(req, identity) {
  const out = {};
  for (const [name, value] of Object.entries(req.headers)) {
    const lower = name.toLowerCase();
    if (HOP_BY_HOP.has(lower) || DROPPED_REQUEST_HEADERS.has(lower)) {
      continue;
    }
    if (lower.startsWith("x-lightseed-")) {
      continue;
    }
    if (lower === "cookie") {
      const stripped = stripSessionCookie(String(value));
      // The PC has no cookies of its own: the sign-in session must not leak
      // into it (or into the PC's request logs).
      if (stripped !== "") {
        out.cookie = stripped;
      }
      continue;
    }
    out[lower] = value;
  }
  out["x-lightseed-proxy-token"] = config.proxyToken;
  out["x-lightseed-user-email"] = identity.email;
  if (identity.sub) {
    out["x-lightseed-user-sub"] = identity.sub;
  }
  return out;
}

function forwardResponseHeaders(upstreamHeaders) {
  const out = {};
  for (const [name, value] of Object.entries(upstreamHeaders)) {
    const lower = name.toLowerCase();
    if (HOP_BY_HOP.has(lower)) {
      continue;
    }
    out[lower] = value;
  }
  return out;
}

// ------------------------------------------------------------------ proxy ---

function parseWebError(body) {
  if (body.length === 0 || body.length > config.offlineSniffBytes) {
    return false;
  }
  let parsed;
  try {
    parsed = JSON.parse(body.toString("utf8"));
  } catch {
    return false;
  }
  return parsed !== null && typeof parsed === "object" &&
    parsed.schema === WEB_ERROR_SCHEMA &&
    typeof parsed.code === "string" &&
    typeof parsed.message === "string";
}

function proxyRequest(req, res, identity) {
  const transport = config.useTailscaleProxy ? "proxy" : "direct";
  const options = requestOptions(req.url, forwardRequestHeaders(req, identity),
    transport);
  options.method = req.method;
  options.agent = agent;

  let headSent = false;
  let finished = false;
  const started = Date.now();

  const upstreamRequest = http.request(options, (upstreamResponse) => {
    clearTimeout(connectTimer);
    const status = upstreamResponse.statusCode || 0;

    if (status === 502 || status === 504) {
      // Classic shape of "the Tailscale proxy could not reach the PC". The PC
      // itself also uses 502/504 for some failures, so sniff the body: only a
      // real qwen_engine.web_error.v1 document is passed through.
      sniffUnreachable(upstreamResponse, req, res, identity, started);
      return;
    }

    headSent = true;
    finished = true;
    // Response headers arrived: the rest of the exchange may take minutes
    // (image generation), so only an idle socket ends it early.
    upstreamRequest.setTimeout(config.idleTimeoutMs, () => {
      upstreamRequest.destroy(new Error("upstream idle timeout"));
    });
    res.writeHead(status, forwardResponseHeaders(upstreamResponse.headers));
    upstreamResponse.pipe(res);
    upstreamResponse.on("error", () => res.destroy());
    upstreamResponse.on("close", () => {
      if (!upstreamResponse.complete) {
        res.destroy();
      }
    });
    res.on("close", () => upstreamResponse.destroy());
    if (config.logRequests) {
      log("upstream", {
        method: req.method,
        path: req.url,
        status,
        ms: Date.now() - started,
        user: userTag(identity.email),
      });
    }
  });

  // The connect timeout covers reaching the PC, not the PC thinking. Once the
  // connection is up, a slow job must not be torn down at 10s: image
  // generation legitimately takes minutes before it sends any header, and
  // answering 503 "offline" to a PC that is merely busy is a lie. So the
  // timer is relaxed to the idle timeout as soon as there is a connected
  // socket, and the idle timeout then governs the whole exchange.
  const connectTimer = setTimeout(() => {
    upstreamRequest.destroy(new Error("upstream connect timeout"));
  }, config.connectTimeoutMs);

  // Relaxed exactly once: the connect timer must not be re-armed under any
  // later socket event (keep-alive handover, retry), or a long request would
  // be killed by a timer that no longer describes anything.
  let relaxed = false;
  const relax = () => {
    if (relaxed) {
      return;
    }
    relaxed = true;
    clearTimeout(connectTimer);
    // Armed before headers arrive, so a PC that accepts the connection and
    // then goes silent still ends at the idle timeout rather than hanging.
    upstreamRequest.setTimeout(config.idleTimeoutMs, () => {
      upstreamRequest.destroy(new Error("upstream idle timeout"));
    });
  };

  upstreamRequest.on("socket", (socket) => {
    // A pooled keep-alive socket is already connected: waiting for a "connect"
    // event that has long since fired would leave the connect timer armed and
    // turn the second and later requests into "offline" at 10s.
    if (socket.connecting) {
      socket.once("connect", relax);
    } else {
      relax();
    }
  });

  upstreamRequest.on("error", (error) => {
    clearTimeout(connectTimer);
    if (finished) {
      res.destroy();
      return;
    }
    finished = true;
    log("upstream unreachable", {
      method: req.method,
      path: req.url,
      reason: error.code || error.message,
      user: userTag(identity.email),
    });
    sendOffline(res);
  });

  // Client went away mid-upload: stop talking to the PC.
  res.on("close", () => {
    if (!finished) {
      upstreamRequest.destroy();
    }
  });

  req.pipe(upstreamRequest);
}

function sniffUnreachable(upstreamResponse, req, res, identity, started) {
  const chunks = [];
  let size = 0;
  let settled = false;

  const forwardWebError = () => {
    if (settled) {
      return;
    }
    settled = true;
    const body = Buffer.concat(chunks);
    const headers = forwardResponseHeaders(upstreamResponse.headers);
    headers["content-length"] = String(body.length);
    headers["cache-control"] = "no-store";
    res.writeHead(upstreamResponse.statusCode, headers);
    res.end(body);
    log("upstream refused", {
      method: req.method,
      path: req.url,
      status: upstreamResponse.statusCode,
      ms: Date.now() - started,
      user: userTag(identity.email),
    });
  };

  const offline = (reason) => {
    if (settled) {
      return;
    }
    settled = true;
    upstreamResponse.destroy();
    log("upstream unreachable", {
      method: req.method,
      path: req.url,
      reason,
      status: upstreamResponse.statusCode,
      user: userTag(identity.email),
    });
    sendOffline(res);
  };

  upstreamResponse.on("data", (chunk) => {
    if (settled) {
      return;
    }
    size += chunk.length;
    if (size > config.offlineSniffBytes) {
      offline("body_too_large_for_web_error");
      return;
    }
    chunks.push(chunk);
  });

  upstreamResponse.on("end", () => {
    if (settled) {
      return;
    }
    if (parseWebError(Buffer.concat(chunks))) {
      forwardWebError();
    } else {
      offline("not_a_web_error_document");
    }
  });

  upstreamResponse.on("error", () => offline("upstream_error"));
}

// ---------------------------------------------------------------- server ---

const agent = new http.Agent({
  keepAlive: true,
  maxSockets: 32,
  maxFreeSockets: 8,
  timeout: 60_000,
});

async function handle(req, res) {
  res.setHeader("x-content-type-options", "nosniff");

  const target = req.url || "";
  if (!target.startsWith("/") || target.startsWith("//")) {
    webError(res, 400, "target_form_unsupported",
      "only origin-form request targets are routed");
    return;
  }

  if ((req.method === "GET" || req.method === "HEAD") &&
      target.split("?")[0] === "/healthz") {
    await handleHealth(req, res);
    return;
  }

  const identity = identityFor(req);
  if (identity.error) {
    log("request without identity", {
      method: req.method,
      path: target,
      peer: req.socket.remoteAddress || "-",
    });
    webError(res, 401, "auth_required", "sign-in is required");
    return;
  }

  const jobSubmission = isJobSubmission(req.method, target);
  const limiter = jobSubmission ? jobLimiter : generalLimiter;
  const retryAfter = limiter.take(identity.email);
  if (retryAfter > 0) {
    log("rate limited", {
      method: req.method,
      path: target,
      scope: jobSubmission ? "job_submission" : "general",
      user: userTag(identity.email),
    });
    webError(res, 429, "rate_limited",
      "too many requests; please slow down and try again",
      { "retry-after": String(retryAfter) });
    return;
  }

  proxyRequest(req, res, identity);
}

const server = http.createServer((req, res) => {
  handle(req, res).catch((error) => {
    log("handler failed", { reason: error && error.message });
    if (!res.headersSent) {
      webError(res, 500, "internal_error", "the gateway failed to handle this request");
    } else {
      res.destroy();
    }
  });
});

// Generation replies can be slow and uploads are up to 32 MiB: no request
// deadline of the runtime's own, and no per-socket inactivity deadline.
server.requestTimeout = 0;
server.timeout = 0;
server.headersTimeout = 60_000;
server.keepAliveTimeout = 65_000;

server.on("clientError", (error, socket) => {
  // Never answer with the raw error text.
  if (socket.writable) {
    socket.end("HTTP/1.1 400 Bad Request\r\ncontent-length: 0\r\n" +
      "connection: close\r\n\r\n");
  } else {
    socket.destroy();
  }
});

server.listen(config.listenPort, config.listenHost, () => {
  log("listening", {
    mode: config.mode,
    address: `${config.listenHost}:${config.listenPort}`,
    upstream: config.upstreamText,
    transport: config.useTailscaleProxy ? "tailscale-proxy" : "direct",
    version: config.version,
  });
});

function shutdown(signal) {
  log("shutting down", { signal });
  server.close(() => {
    agent.destroy();
    process.exit(0);
  });
  // In-flight requests get a moment to finish; the supervisor kills the rest.
  setTimeout(() => process.exit(0), 5_000).unref();
}

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));
