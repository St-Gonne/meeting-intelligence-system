#!/usr/bin/env node

// Minimal dependency-free obs-websocket 5.x request client.
// The password is accepted only through MI_OBS_WS_PASSWORD.

import { createHash, randomUUID } from "node:crypto";

const endpoint = process.env.MI_OBS_WS_URL || "ws://127.0.0.1:4455";
const password = process.env.MI_OBS_WS_PASSWORD || "";
const requestType = process.argv[2] || "GetVersion";
const requestDataText = process.argv[3] || "{}";

let requestData;
try {
  requestData = JSON.parse(requestDataText);
} catch {
  console.error("request data must be valid JSON");
  process.exit(2);
}

if (!requestData || Array.isArray(requestData) || typeof requestData !== "object") {
  console.error("request data must be a JSON object");
  process.exit(2);
}

const requestId = randomUUID();
const socket = new WebSocket(endpoint);
const timeout = setTimeout(() => {
  console.error("OBS WebSocket request timed out");
  socket.close();
  process.exit(3);
}, 8000);

function sha256Base64(value) {
  return createHash("sha256").update(value, "utf8").digest("base64");
}

function authenticationValue(challenge, salt) {
  const secret = sha256Base64(`${password}${salt}`);
  return sha256Base64(`${secret}${challenge}`);
}

function send(payload) {
  socket.send(JSON.stringify(payload));
}

socket.addEventListener("error", () => {
  clearTimeout(timeout);
  console.error("OBS WebSocket connection failed");
  process.exit(4);
});
socket.addEventListener("message", (event) => {
  let message;
  try {
    message = JSON.parse(String(event.data));
  } catch {
    clearTimeout(timeout);
    console.error("OBS WebSocket returned invalid JSON");
    process.exit(5);
  }

  if (message.op === 0) {
    const identify = { rpcVersion: 1, eventSubscriptions: 0 };
    const authentication = message.d?.authentication;
    if (authentication) {
      if (!password) {
        clearTimeout(timeout);
        console.error("OBS WebSocket requires MI_OBS_WS_PASSWORD");
        process.exit(6);
      }
      identify.authentication = authenticationValue(
        authentication.challenge,
        authentication.salt,
      );
    }
    send({ op: 1, d: identify });
    return;
  }

  if (message.op === 2) {
    send({
      op: 6,
      d: {
        requestType,
        requestId,
        requestData,
      },
    });
    return;
  }

  if (message.op === 7 && message.d?.requestId === requestId) {
    clearTimeout(timeout);
    const status = message.d.requestStatus || {};
    const output = {
      requestType,
      ok: Boolean(status.result),
      code: status.code,
      comment: status.comment || null,
      responseData: message.d.responseData || {},
    };
    console.log(JSON.stringify(output));
    socket.close();
    process.exit(output.ok ? 0 : 7);
  }
});
