"use strict";

// Tests for the Railway edge gateway (deploy/railway/gateway.js).
//
// Every test runs the real gateway as a child process in dev mode against an
// in-process fake upstream, so what is asserted is the shipped code path
// (request rewriting, streaming, offline fallback, rate limiting, health),
// not a copy of it.
//
//   node --test test/
//
// No dependencies, no network: loopback only, ephemeral ports.

const test = require("node:test");
const assert = require("node:assert/strict");
const http = require("node:http");
const net = require("node:net");
const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawn } = require("node:child_process");

const GATEWAY = path.join(__dirname, "..", "gateway.js");
const LOGO = fs.readFileSync(
  path.join(__dirname, "..", "assets", "lightseed-logo-on-dark.svg"), "utf8");
const PROXY_TOKEN = "test-proxy-token-0123456789";

// ---------------------------------------------------------------- helpers ---

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

function listen(server) {
  return new Promise((resolve, reject) => {
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => resolve(server.address().port));
  });
}

function close(server) {
  return new Promise((resolve) => server.close(() => resolve()));
}

function request(port, options, body) {
  return new Promise((resolve, reject) => {
    const req = http.request({
      host: "127.0.0.1",
      port,
      method: options.method || "GET",
      path: options.path || "/",
      headers: options.headers || {},
    }, (res) => {
      const chunks = [];
      res.on("data", (chunk) => chunks.push(chunk));
      res.on("end", () => resolve({
        status: res.statusCode,
        headers: res.headers,
        body: Buffer.concat(chunks),
        text: Buffer.concat(chunks).toString("utf8"),
      }));
      res.on("error", reject);
    });
    req.on("error", reject);
    req.end(body);
  });
}

// Starts the gateway in dev mode and waits for its listening line.
async function startGateway(env) {
  const port = await freePort();
  const child = spawn(process.execPath, [GATEWAY], {
    env: {
      PATH: process.env.PATH,
      LIGHTSEED_MODE: "dev",
      PORT: String(port),
      // Proxy mode listens on this loopback port; the tests always talk to
      // `port`, so keep the two identical unless a case overrides it.
      LIGHTSEED_GATEWAY_PORT: String(port),
      LIGHTSEED_NO_TAILSCALE: "1",
      LIGHTSEED_PROXY_TOKEN: PROXY_TOKEN,
      LIGHTSEED_DEV_EMAIL: "Invited.Person@Example.COM",
      LIGHTSEED_DEV_SUB: "1234567890",
      LIGHTSEED_VERSION: "abc1234",
      ...env,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  let stderr = "";
  child.stderr.on("data", (chunk) => { stderr += chunk.toString(); });

  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`gateway did not start; stderr:\n${stderr}`));
    }, 10_000);
    const poll = () => {
      if (stderr.includes("listening")) {
        clearTimeout(timer);
        resolve();
        return;
      }
      if (child.exitCode !== null) {
        clearTimeout(timer);
        reject(new Error(`gateway exited; stderr:\n${stderr}`));
        return;
      }
      setTimeout(poll, 25);
    };
    poll();
  });

  return {
    port,
    child,
    stderr: () => stderr,
    stop: () => new Promise((resolve) => {
      if (child.exitCode !== null) {
        resolve();
        return;
      }
      child.once("exit", () => resolve());
      child.kill("SIGTERM");
    }),
  };
}

// Fake PC: records what it was sent and answers from a per-test handler.
async function startUpstream(handler) {
  const observed = [];
  const server = http.createServer((req, res) => {
    const record = { method: req.method, url: req.url, headers: req.headers };
    observed.push(record);
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => {
      record.body = Buffer.concat(chunks);
      handler(req, res, record);
    });
  });
  const port = await listen(server);
  return {
    port,
    observed,
    stop: () => new Promise((resolve) => {
      server.closeAllConnections?.();
      close(server).then(resolve);
    }),
  };
}

function jsonUpstream(status, payload) {
  return (req, res) => {
    const body = Buffer.from(JSON.stringify(payload), "utf8");
    res.writeHead(status, {
      "content-type": "application/json; charset=utf-8",
      "content-length": String(body.length),
    });
    res.end(body);
  };
}

// ------------------------------------------------------- header rewriting ---

test("strips client X-Lightseed-* headers and injects identity", async () => {
  const upstream = await startUpstream(jsonUpstream(200, { ok: true }));
  const gateway = await startGateway({
    LIGHTSEED_UPSTREAM: `127.0.0.1:${upstream.port}`,
  });
  try {
    const response = await request(gateway.port, {
      method: "GET",
      path: "/api/v1/status",
      headers: {
        "X-Lightseed-Proxy-Token": "forged-token",
        "X-Lightseed-User-Email": "attacker@evil.example",
        "X-Lightseed-User-Sub": "666",
        "X-Lightseed-Anything-Else": "nope",
        "X-Forwarded-Email": "attacker@evil.example",
        "X-Forwarded-User": "666",
        "X-Qwen-UI": "1",
        Origin: "https://lightseed.example",
        Accept: "application/json",
      },
    });
    assert.equal(response.status, 200);

    const seen = upstream.observed.at(-1).headers;
    assert.equal(seen["x-lightseed-proxy-token"], PROXY_TOKEN);
    // Dev identity, lowercased.
    assert.equal(seen["x-lightseed-user-email"], "invited.person@example.com");
    assert.equal(seen["x-lightseed-user-sub"], "1234567890");
    assert.equal(seen["x-lightseed-anything-else"], undefined);
    // Proxy trust headers from the client never reach the PC.
    assert.equal(seen["x-forwarded-email"], undefined);
    assert.equal(seen["x-forwarded-user"], undefined);
    // Everything else passes through untouched, Origin included.
    assert.equal(seen.origin, "https://lightseed.example");
    assert.equal(seen["x-qwen-ui"], "1");
    assert.equal(seen.accept, "application/json");
  } finally {
    await gateway.stop();
    await upstream.stop();
  }
});

test("drops the sign-in session cookie but keeps the other cookies", async () => {
  const upstream = await startUpstream(jsonUpstream(200, {}));
  const gateway = await startGateway({
    LIGHTSEED_UPSTREAM: `127.0.0.1:${upstream.port}`,
  });
  try {
    await request(gateway.port, {
      path: "/",
      headers: {
        Cookie: "_lightseed_session=secret-session; qwen_theme=neon",
      },
    });
    const cookie = upstream.observed.at(-1).headers.cookie;
    assert.equal(cookie, "qwen_theme=neon");
  } finally {
    await gateway.stop();
    await upstream.stop();
  }
});

test("requires an identity header in proxy mode", async () => {
  const upstream = await startUpstream(jsonUpstream(200, {}));
  const gateway = await startGateway({
    LIGHTSEED_MODE: "proxy",
    LIGHTSEED_UPSTREAM: `127.0.0.1:${upstream.port}`,
    LIGHTSEED_DEV_EMAIL: "",
  });
  try {
    const anonymous = await request(gateway.port, { path: "/api/v1/status" });
    assert.equal(anonymous.status, 401);
    assert.equal(JSON.parse(anonymous.text).code, "auth_required");
    assert.equal(upstream.observed.length, 0);

    const signedIn = await request(gateway.port, {
      path: "/api/v1/status",
      headers: {
        "X-Forwarded-Email": "Someone@Example.com",
        "X-Forwarded-User": "42",
      },
    });
    assert.equal(signedIn.status, 200);
    const seen = upstream.observed.at(-1).headers;
    assert.equal(seen["x-lightseed-user-email"], "someone@example.com");
    assert.equal(seen["x-lightseed-user-sub"], "42");
  } finally {
    await gateway.stop();
    await upstream.stop();
  }
});

test("rejects a malformed forwarded email instead of trusting it", async () => {
  const upstream = await startUpstream(jsonUpstream(200, {}));
  const gateway = await startGateway({
    LIGHTSEED_MODE: "proxy",
    LIGHTSEED_UPSTREAM: `127.0.0.1:${upstream.port}`,
    LIGHTSEED_DEV_EMAIL: "",
  });
  try {
    for (const bad of ["", "no-at-sign", "two@at@example.com", "a@b",
      "trailing@example.com.", "spaced @example.com"]) {
      const response = await request(gateway.port, {
        path: "/api/v1/status",
        headers: { "X-Forwarded-Email": bad },
      });
      assert.equal(response.status, 401, `email ${JSON.stringify(bad)}`);
    }
    assert.equal(upstream.observed.length, 0);
  } finally {
    await gateway.stop();
    await upstream.stop();
  }
});

// ------------------------------------------------------------- streaming ---

test("streams the response without buffering it", async () => {
  const upstream = await startUpstream((req, res) => {
    res.writeHead(200, { "content-type": "text/plain" });
    res.write("first-chunk\n");
    setTimeout(() => {
      res.end("second-chunk\n");
    }, 400);
  });
  const gateway = await startGateway({
    LIGHTSEED_UPSTREAM: `127.0.0.1:${upstream.port}`,
  });
  try {
    const started = Date.now();
    const timings = await new Promise((resolve, reject) => {
      const req = http.request({
        host: "127.0.0.1",
        port: gateway.port,
        path: "/stream",
      }, (res) => {
        let first = 0;
        let last = 0;
        const chunks = [];
        res.on("data", (chunk) => {
          if (first === 0) {
            first = Date.now() - started;
          }
          last = Date.now() - started;
          chunks.push(chunk);
        });
        res.on("end", () => resolve({
          first,
          last,
          text: Buffer.concat(chunks).toString("utf8"),
        }));
        res.on("error", reject);
      });
      req.on("error", reject);
      req.end();
    });
    assert.equal(timings.text, "first-chunk\nsecond-chunk\n");
    // The first byte has to arrive while the upstream is still holding the
    // socket open; a buffering proxy would deliver both at the end.
    assert.ok(timings.first < 250, `first byte at ${timings.first}ms`);
    assert.ok(timings.last - timings.first >= 250,
      `gap ${timings.last - timings.first}ms`);
  } finally {
    await gateway.stop();
    await upstream.stop();
  }
});

test("streams a 40 MiB upload through unchanged", async () => {
  const upstream = await startUpstream((req, res, record) => {
    const digest = crypto.createHash("sha256").update(record.body)
      .digest("hex");
    jsonUpstream(200, { bytes: record.body.length, sha256: digest })(req, res);
  });
  const gateway = await startGateway({
    LIGHTSEED_UPSTREAM: `127.0.0.1:${upstream.port}`,
  });
  try {
    const payload = crypto.randomBytes(40 * 1024 * 1024);
    const expected = crypto.createHash("sha256").update(payload).digest("hex");
    const response = await request(gateway.port, {
      method: "POST",
      path: "/api/v1/references",
      headers: {
        "content-type": "application/octet-stream",
        "content-length": String(payload.length),
        "X-Qwen-UI": "1",
      },
    }, payload);
    assert.equal(response.status, 200);
    const echoed = JSON.parse(response.text);
    assert.equal(echoed.bytes, payload.length);
    assert.equal(echoed.sha256, expected);
  } finally {
    await gateway.stop();
    await upstream.stop();
  }
});

test("does not call a busy PC offline while it is still thinking", async () => {
  // Regression: the connect timeout used to stay armed after the socket
  // connected, so any request whose headers took longer than the connect
  // timeout (10s by default) was answered with the 503 offline page even
  // though the PC was up and working. Image generation regularly takes
  // minutes to produce a header.
  const upstream = await startUpstream((req, res) => {
    setTimeout(() => jsonUpstream(200, { ok: true })(req, res), 1_500);
  });
  const gateway = await startGateway({
    LIGHTSEED_UPSTREAM: `127.0.0.1:${upstream.port}`,
    // Well below the upstream's delay: if the connect timer survives the
    // connection, this request dies at 300ms with the offline page.
    LIGHTSEED_CONNECT_TIMEOUT_MS: "300",
  });
  try {
    const response = await request(gateway.port, { path: "/api/v1/status" });
    assert.equal(response.status, 200);
    assert.equal(JSON.parse(response.text).ok, true);
  } finally {
    await gateway.stop();
    await upstream.stop();
  }
});

test("still gives up on a PC that connects and then goes silent", async () => {
  // The relaxed timer must not become "wait forever": a socket that accepts
  // and never sends anything has to end at the idle timeout.
  const silent = net.createServer((socket) => socket.on("data", () => {}));
  await listen(silent);
  const gateway = await startGateway({
    LIGHTSEED_UPSTREAM: `127.0.0.1:${silent.address().port}`,
    LIGHTSEED_UPSTREAM_IDLE_TIMEOUT_MS: "400",
  });
  try {
    const started = Date.now();
    const response = await request(gateway.port, { path: "/api/v1/status" });
    const elapsed = Date.now() - started;
    assert.equal(response.status, 503);
    assert.ok(elapsed >= 350, `ended too early after ${elapsed}ms`);
    assert.ok(elapsed < 5_000, `idle timeout did not fire, ${elapsed}ms`);
  } finally {
    await gateway.stop();
    await close(silent);
  }
});

test("repeated slow requests on pooled sockets all succeed", async () => {
  // Regression: the connect timer was relaxed from a "connect" listener. A
  // keep-alive socket is already connected when it is handed back for the
  // next request, so that event never fires again and every request after the
  // first was killed at the connect timeout -- the real production shape,
  // because the gateway pools sockets.
  const upstream = await startUpstream((req, res) => {
    setTimeout(() => jsonUpstream(200, { ok: true })(req, res), 700);
  });
  const gateway = await startGateway({
    LIGHTSEED_UPSTREAM: `127.0.0.1:${upstream.port}`,
    LIGHTSEED_CONNECT_TIMEOUT_MS: "300",
  });
  try {
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      const response = await request(gateway.port, { path: "/api/v1/status" });
      assert.equal(response.status, 200, `attempt ${attempt} failed`);
    }
  } finally {
    await gateway.stop();
    await upstream.stop();
  }
});

// ---------------------------------------------------------- offline page ---

test("serves the branded offline page when the PC is unreachable", async () => {
  const deadPort = await freePort();
  const gateway = await startGateway({
    LIGHTSEED_UPSTREAM: `127.0.0.1:${deadPort}`,
  });
  try {
    const response = await request(gateway.port, { path: "/api/v1/status" });
    assert.equal(response.status, 503);
    assert.match(response.headers["content-type"], /text\/html/);
    assert.equal(response.headers["x-lightseed-offline"], "1");
    assert.ok(response.text.includes("Lightseed Studio is offline"));
    assert.ok(response.text.includes("switched off"));
    // Brand logo inlined, version stamped.
    assert.ok(response.text.includes('aria-label="Lightseed"'));
    assert.ok(response.text.includes("<code>abc1234</code>"));
    // Never a raw proxy error.
    assert.ok(!/ECONNREFUSED|AggregateError|socket hang up/.test(response.text));
  } finally {
    await gateway.stop();
  }
});

test("passes a real upstream 502 web error through, but not a blank one", async () => {
  let mode = "web-error";
  const upstream = await startUpstream((req, res) => {
    if (mode === "web-error") {
      jsonUpstream(502, {
        schema: "qwen_engine.web_error.v1",
        code: "generation_failed",
        message: "the diffusion process exited",
      })(req, res);
      return;
    }
    res.writeHead(502, { "content-type": "text/html" });
    res.end("<html><body>proxy error</body></html>");
  });
  const gateway = await startGateway({
    LIGHTSEED_UPSTREAM: `127.0.0.1:${upstream.port}`,
  });
  try {
    const passthrough = await request(gateway.port, { path: "/api/v1/jobs/x" });
    assert.equal(passthrough.status, 502);
    const parsed = JSON.parse(passthrough.text);
    assert.equal(parsed.schema, "qwen_engine.web_error.v1");
    assert.equal(parsed.code, "generation_failed");

    mode = "blank";
    const replaced = await request(gateway.port, { path: "/api/v1/jobs/x" });
    assert.equal(replaced.status, 503);
    assert.equal(replaced.headers["x-lightseed-offline"], "1");
    assert.ok(!replaced.text.includes("proxy error"));
  } finally {
    await gateway.stop();
    await upstream.stop();
  }
});

// ----------------------------------------------------------- rate limits ---

test("rate limits per identity, tighter for job submission", async () => {
  const upstream = await startUpstream(jsonUpstream(200, {}));
  const gateway = await startGateway({
    LIGHTSEED_UPSTREAM: `127.0.0.1:${upstream.port}`,
    LIGHTSEED_RATE_LIMIT_MAX: "30",
    LIGHTSEED_JOB_RATE_LIMIT_MAX: "5",
  });
  try {
    for (let i = 0; i < 5; i += 1) {
      const ok = await request(gateway.port, {
        method: "POST",
        path: "/api/v1/jobs",
        headers: { "content-type": "application/json", "X-Qwen-UI": "1" },
      }, "{}");
      assert.equal(ok.status, 200, `job ${i + 1}`);
    }
    const limited = await request(gateway.port, {
      method: "POST",
      path: "/api/v1/jobs",
      headers: { "content-type": "application/json", "X-Qwen-UI": "1" },
    }, "{}");
    assert.equal(limited.status, 429);
    const error = JSON.parse(limited.text);
    assert.equal(error.code, "rate_limited");
    assert.equal(error.schema, "qwen_engine.web_error.v1");
    assert.ok(Number(limited.headers["retry-after"]) >= 1);

    // Reading jobs is a different bucket, still open.
    const read = await request(gateway.port, { path: "/api/v1/jobs" });
    assert.equal(read.status, 200);
  } finally {
    await gateway.stop();
    await upstream.stop();
  }
});

// ---------------------------------------------------------------- health ---

test("reports health without leaking addresses", async () => {
  const upstream = await startUpstream(jsonUpstream(200, { ok: true }));
  const gateway = await startGateway({
    LIGHTSEED_UPSTREAM: `127.0.0.1:${upstream.port}`,
  });
  try {
    const healthy = await request(gateway.port, { path: "/healthz" });
    assert.equal(healthy.status, 200);
    const body = JSON.parse(healthy.text);
    assert.equal(body.status, "ok");
    assert.equal(body.upstream, "reachable");
    assert.ok(!healthy.text.includes("127.0.0.1"));
    assert.ok(!healthy.text.includes(String(upstream.port)));
    assert.ok(!healthy.text.includes("100."));
  } finally {
    await gateway.stop();
    await upstream.stop();
  }

  const deadPort = await freePort();
  const offline = await startGateway({
    LIGHTSEED_UPSTREAM: `127.0.0.1:${deadPort}`,
    LIGHTSEED_HEALTH_CACHE_MS: "0",
  });
  try {
    const degraded = await request(offline.port, { path: "/healthz" });
    // A switched-off PC is not a broken container: stay 200 so Railway does
    // not restart the service forever.
    assert.equal(degraded.status, 200);
    const body = JSON.parse(degraded.text);
    assert.equal(body.status, "degraded");
    assert.equal(body.upstream, "unreachable");
    assert.ok(!degraded.text.includes(String(deadPort)));
  } finally {
    await offline.stop();
  }
});

test("healthz answers 503 when tailscaled itself is down", async () => {
  const upstream = await startUpstream(jsonUpstream(200, {}));
  const gateway = await startGateway({
    // Speak to Tailscale (not LIGHTSEED_NO_TAILSCALE), but point both the
    // proxy and its control socket at ports with nothing behind them.
    LIGHTSEED_NO_TAILSCALE: "",
    LIGHTSEED_UPSTREAM: `127.0.0.1:${upstream.port}`,
    LIGHTSEED_TAILSCALE_PROXY: `127.0.0.1:${await freePort()}`,
    LIGHTSEED_TAILSCALED_SOCKET: path.join(os.tmpdir(),
      `lightseed-missing-${process.pid}.sock`),
    LIGHTSEED_HEALTH_CACHE_MS: "0",
  });
  try {
    const response = await request(gateway.port, { path: "/healthz" });
    assert.equal(response.status, 503);
    const body = JSON.parse(response.text);
    assert.equal(body.status, "tailscale_down");
  } finally {
    await gateway.stop();
    await upstream.stop();
  }
});

// ------------------------------------------------------------- bad input ---

test("refuses absolute-form request targets", async () => {
  const upstream = await startUpstream(jsonUpstream(200, {}));
  const gateway = await startGateway({
    LIGHTSEED_UPSTREAM: `127.0.0.1:${upstream.port}`,
  });
  try {
    // Raw socket: node's http client would normalise the target away.
    const raw = await new Promise((resolve, reject) => {
      const socket = net.connect(gateway.port, "127.0.0.1", () => {
        socket.write("GET http://evil.example/api/v1/status HTTP/1.1\r\n" +
          "Host: evil.example\r\nConnection: close\r\n\r\n");
      });
      let data = "";
      socket.on("data", (chunk) => { data += chunk.toString(); });
      socket.on("end", () => resolve(data));
      socket.on("error", reject);
    });
    assert.match(raw, /^HTTP\/1\.1 400/);
    assert.equal(upstream.observed.length, 0);
  } finally {
    await gateway.stop();
    await upstream.stop();
  }
});

test("flags a missing upstream configuration at startup", async () => {
  const child = spawn(process.execPath, [GATEWAY], {
    env: { PATH: process.env.PATH, LIGHTSEED_MODE: "dev", PORT: "1" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stderr = "";
  child.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
  const code = await new Promise((resolve) => child.on("exit", resolve));
  assert.notEqual(code, 0);
  assert.match(stderr, /LIGHTSEED_UPSTREAM is required/);
});

// ---------------------------------------------------------------- assets ---

test("brand logo is a copy of the shipped app logo", () => {
  const app = path.join(__dirname, "..", "..", "..", "src", "qwen_engine",
    "web_static", "brand", "lightseed-logo-on-dark.svg");
  assert.ok(LOGO.length > 1000);
  assert.equal(LOGO, fs.readFileSync(app, "utf8"));
});

// ------------------------------------------------------- shipped oauth2 args ---

// These pin the two mistakes that make sign-in "work" locally and fail in
// production, both of which were found by running the real binary:
//   * oauth2-proxy's config key for --upstream is "upstreams" (plural). Setting
//     OAUTH2_PROXY_UPSTREAM is accepted silently and ignored, so every
//     authenticated request returns oauth2-proxy's own bare 404.
//   * "--scope=openid email" is ONE argument. Split on white space, "email"
//     becomes a separate argv entry and is dropped, so Google returns an ID
//     token with no email claim.
const RAILWAY_DIR = path.join(__dirname, "..");

function shippedArgs() {
  return fs.readFileSync(path.join(RAILWAY_DIR, "oauth2-proxy.args"), "utf8")
    .split("\n")
    .map((line) => line.trimEnd())
    .filter((line) => line !== "" && !line.trimStart().startsWith("#"));
}

test("ships the email scope as a single argument", () => {
  const scope = shippedArgs().filter((arg) => arg.startsWith("--scope="));
  assert.deepEqual(scope, ["--scope=openid email"]);
  // And nothing else may smuggle a bare "email" word in as its own argument.
  assert.ok(!shippedArgs().includes("email"),
    "a bare 'email' argument means the scope line was split");
});

test("keeps the sign-in boundary closed: no trusted-proxy header trust", () => {
  const args = shippedArgs();
  assert.ok(!args.some((arg) => arg.startsWith("--reverse-proxy")),
    "reverse-proxy makes oauth2-proxy trust client X-Forwarded-* headers");
  assert.ok(!args.some((arg) => arg.startsWith("--trusted-proxy-ip")),
    "no CIDR list may be trusted to supply identity headers");
  assert.ok(args.includes("--cookie-secure=true"),
    "the session cookie must be https-only");
  assert.ok(args.includes("--skip-auth-route=^/healthz$"),
    "the health probe must bypass sign-in or Railway restart-loops");
});

test("names the upstream variable in the plural, as oauth2-proxy requires", () => {
  const entrypoint = fs.readFileSync(path.join(RAILWAY_DIR, "entrypoint.sh"),
    "utf8");
  const assigned = entrypoint.match(/export OAUTH2_PROXY_UPSTREAM\w*=/g) || [];
  assert.deepEqual(assigned, ["export OAUTH2_PROXY_UPSTREAMS="],
    "the singular OAUTH2_PROXY_UPSTREAM is silently ignored by oauth2-proxy");
});

// ------------------------------------------------------- tailscale transport ---

test("dials the PC through the Tailscale HTTP proxy, absolute-form", async () => {
  // The userspace proxy is a plain HTTP proxy: the request line has to carry
  // the full URL, or tailscaled answers 400 and every request looks offline.
  const upstream = await startUpstream(jsonUpstream(200, { ok: true }));
  const seenTargets = [];
  const proxy = http.createServer((req, res) => {
    seenTargets.push(req.url);
    // Forward by hand so the test does not depend on the gateway's own socket
    // handling: only the target form is under test here.
    const target = new URL(req.url);
    const forwarded = http.request({
      host: target.hostname,
      port: target.port,
      path: target.pathname + target.search,
      method: req.method,
      headers: { ...req.headers, host: target.host },
    }, (upstreamResponse) => {
      res.writeHead(upstreamResponse.statusCode, upstreamResponse.headers);
      upstreamResponse.pipe(res);
    });
    forwarded.on("error", () => {
      res.writeHead(502, { "content-length": "0" });
      res.end();
    });
    req.pipe(forwarded);
  });
  const proxyPort = await listen(proxy);

  const gateway = await startGateway({
    LIGHTSEED_NO_TAILSCALE: "",
    LIGHTSEED_UPSTREAM: `127.0.0.1:${upstream.port}`,
    LIGHTSEED_TAILSCALE_PROXY: `127.0.0.1:${proxyPort}`,
  });
  try {
    const response = await request(gateway.port, {
      path: "/api/v1/status?x=1",
    });
    assert.equal(response.status, 200);
    assert.equal(response.text, '{"ok":true}');
    assert.deepEqual(seenTargets, [
      `http://127.0.0.1:${upstream.port}/api/v1/status?x=1`,
    ]);
    // The PC sees its own address in Host, not the public domain.
    assert.equal(upstream.observed.at(-1).headers.host,
      `127.0.0.1:${upstream.port}`);
  } finally {
    await gateway.stop();
    await upstream.stop();
    await close(proxy);
  }
});

test("the Tailscale proxy answering 502 becomes the offline page", async () => {
  // Exactly what tailscaled does for an unreachable or logged-out node: 502,
  // empty body. A visitor must not see that.
  const proxy = http.createServer((req, res) => {
    res.writeHead(502, { "content-length": "0" });
    res.end();
  });
  const proxyPort = await listen(proxy);
  const gateway = await startGateway({
    LIGHTSEED_NO_TAILSCALE: "",
    LIGHTSEED_UPSTREAM: "100.98.64.62:8766",
    LIGHTSEED_TAILSCALE_PROXY: `127.0.0.1:${proxyPort}`,
  });
  try {
    const response = await request(gateway.port, { path: "/api/v1/status" });
    assert.equal(response.status, 503);
    assert.equal(response.headers["x-lightseed-offline"], "1");
    assert.ok(response.text.includes("Lightseed Studio is offline"));
  } finally {
    await gateway.stop();
    await close(proxy);
  }
});
