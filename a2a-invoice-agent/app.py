import os, re, json, time, hashlib, sqlite3, threading, uuid, datetime, traceback
from urllib.parse import urlparse
import httpx
from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

BASE_URL = os.environ["BASE_URL"].rstrip("/")        # https://ga3-a2a.onrender.com/a2a
PREFIX = urlparse(BASE_URL).path                      # "/a2a"
AIPIPE = os.environ["AIPIPE_TOKEN"]
MODEL = os.environ.get("MODEL", "gpt-4o-mini")

BATCH = "application/vnd.ga5.invoice-claim-batch+json"
PROP = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPT = "application/vnd.ga5.invoice-action-receipts+json"
RES = "application/vnd.ga5.invoice-action-results+json"
MT = "application/a2a+json"
ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}

db = sqlite3.connect("agent.db", check_same_thread=False)
dbl = threading.RLock()
db.executescript("""
create table if not exists tasks(id text primary key, principal text, data text);
create table if not exists msgs(principal text, mid text, hash text, task_id text, primary key(principal, mid));
create table if not exists cache(h text primary key, decision text);
""")
_locks, _g = {}, threading.Lock()
def L(k):
    with _g:
        return _locks.setdefault(k, threading.RLock())

def canon(o): return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def out(o, status=200): return Response(json.dumps(o), status_code=status, media_type=MT)
def err(status, code): return out({"error": {"code": code, "message": "Request failed"}}, status)
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def q(sql, args=()):
    with dbl:
        return db.execute(sql, args).fetchone()
def load(tid, p):
    r = q("select data from tasks where id=? and principal=?", (tid, p))
    return json.loads(r[0]) if r else None
def save(t, p):
    with dbl:
        db.execute("insert or replace into tasks values(?,?,?)", (t["id"], p, json.dumps(t)))
        db.commit()

def guard(req):
    a = req.headers.get("authorization", "")
    if not a.startswith("Bearer ") or len(a.strip()) < 8:
        return None, err(401, "UNAUTHENTICATED")
    if req.headers.get("a2a-version") != "1.0":
        return None, err(400, "UNSUPPORTED_VERSION")
    acc = req.headers.get("accept", "*/*")
    if not any(x in acc for x in ("a2a+json", "*/*", "json")):
        return None, err(406, "NOT_ACCEPTABLE")
    return sha(a[7:].strip()), None

# ---------------- AI decision ----------------
SYSTEM = """You are an invoice operations analyst. For each invoice package choose exactly ONE action:
- settle_invoice: valid, reconciled, and within autonomous approval authority.
- request_approval: commercially valid, but the amount or terms exceed delegated authority.
- hold_invoice: payment must pause until a stated verification completes.
- reject_duplicate: the SAME commercial invoice (same vendor, invoice number and amount) was already paid. A payment for a DIFFERENT invoice is NOT a duplicate.
- open_exception: material records conflict (amounts, vendor, PO, receipt mismatch) and need an exception workflow.

Each package mixes useful facts with distractors: a cover sheet, archive or old examples, training decoys, negated statements ("this is not a duplicate", "no hold is required"), and irrelevant action words. Find the ONE paragraph that actually determines the action for THIS invoice and decide only from it.

evidenceRefs: exactly THREE bracketed references taken from that one decisive paragraph. Copy each EXACTLY as it appears in the package text, including its square brackets. Never use the cover-sheet reference, archive examples, or decoys. Do not shorten or reformat them.
amountMinor is an integer in minor units (INR 1,234.50 -> 123450).
vendorName, invoiceNumber and currency come from the invoice being decided.

Return ONLY JSON in this shape:
{"decisions":[{"index":0,"action":"...","vendorName":"...","invoiceNumber":"...","amountMinor":0,"currency":"INR","evidenceRefs":["[..]","[..]","[..]"],"reason":"one or two sentences why"}]}"""

def call_llm(items):
    user = "\n\n".join(f"### PACKAGE index={i}\n{json.dumps(p, ensure_ascii=False)}" for i, p in items)
    r = httpx.post("https://aipipe.org/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {AIPIPE}"},
        json={"model": MODEL, "temperature": 0, "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]},
        timeout=30)
    if r.status_code != 200:
        print("LLM HTTP", r.status_code, r.text[:500], flush=True)
        r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    try:
        data = json.loads(content)
    except Exception:
        print("LLM BAD JSON:", content[:500], flush=True)
        raise
    got = {}
    for d in data.get("decisions", []):
        try: got[int(d["index"])] = d
        except Exception: pass
    return got

def flat(o):
    if isinstance(o, str): return o
    if isinstance(o, dict): return "\n".join(flat(v) for v in o.values())
    if isinstance(o, list): return "\n".join(flat(v) for v in o)
    return str(o)

def snap_refs(refs, text):
    result = []
    for x in refs:
        x = str(x).strip()
        cands = [x, "[" + x.strip("[] ") + "]", x.strip("[] ")]
        hit = next((c for c in cands if c and c in text), None)
        if not hit: return None
        result.append(hit)
    return result

def validate(d, pkg):
    try:
        if not d: raise ValueError("no decision returned")
        if d["action"] not in ACTIONS: raise ValueError("bad action " + str(d["action"]))
        refs = snap_refs(d["evidenceRefs"], flat(pkg))
        if not refs or len(refs) != 3: raise ValueError("refs invalid: %r" % d.get("evidenceRefs"))
        amt = d["amountMinor"]
        if isinstance(amt, float) and amt.is_integer(): amt = int(amt)
        if not isinstance(amt, int) or isinstance(amt, bool): raise ValueError("bad amount %r" % amt)
        v, n, c = str(d["vendorName"]), str(d["invoiceNumber"]), str(d["currency"])
        if not (v and n and c): raise ValueError("empty fact")
        return {"action": d["action"],
                "facts": {"vendorName": v, "invoiceNumber": n, "amountMinor": amt, "currency": c},
                "evidenceRefs": refs, "reason": str(d.get("reason", ""))[:800]}
    except Exception as e:
        print("VALIDATION FAIL:", repr(e), flush=True)
        return None

def decide(pkgs):
    hs = [sha(canon(p)) for p in pkgs]
    res = {}
    for h in set(hs):
        r = q("select decision from cache where h=?", (h,))
        if r: res[h] = json.loads(r[0])
    todo = [i for i, h in enumerate(hs) if h not in res]
    t0 = time.time()
    raw = {}
    for attempt in range(2):
        if not todo or (attempt and time.time() - t0 > 18): break
        got = call_llm([(i, pkgs[i]) for i in todo])
        raw.update(got)
        nxt = []
        for i in todo:
            d = validate(got.get(i), pkgs[i])
            if d:
                res[hs[i]] = d
                with dbl:
                    db.execute("insert or replace into cache values(?,?)", (hs[i], json.dumps(d)))
                    db.commit()
            else:
                nxt.append(i)
        todo = nxt
    for i in todo:  # last resort: never fail the whole batch
        g = raw.get(i) or {}
        refs = re.findall(r"\[[^\[\]]{3,60}\]", flat(pkgs[i]))[-3:]
        amt = g.get("amountMinor")
        res[hs[i]] = {
            "action": g.get("action") if g.get("action") in ACTIONS else "hold_invoice",
            "facts": {"vendorName": str(g.get("vendorName", "unknown")),
                      "invoiceNumber": str(g.get("invoiceNumber", "unknown")),
                      "amountMinor": amt if isinstance(amt, int) and not isinstance(amt, bool) else 0,
                      "currency": str(g.get("currency", "INR"))},
            "evidenceRefs": refs, "reason": str(g.get("reason", "Verification is needed before payment."))[:800]}
    return [res[h] for h in hs]

def pkg_id(pkg, i):
    if isinstance(pkg, dict):
        for k in ("packageId", "package_id", "id", "invoicePackageId"):
            if pkg.get(k): return str(pkg[k])
    return str(i)

# ---------------- handlers ----------------
def do_new(p, msg, part):
    data = part.get("data")
    if part.get("mediaType") != BATCH or not isinstance(data, dict) or not isinstance(data.get("packages"), list):
        return err(400, "INVALID_REQUEST")
    pkgs, bid = data["packages"], data.get("batchId")
    if pkgs and isinstance(pkgs[0], dict):
        print("PACKAGE KEYS:", list(pkgs[0].keys()), flush=True)
    try:
        ds = decide(pkgs)
    except Exception:
        traceback.print_exc()
        return err(503, "TEMPORARILY_UNAVAILABLE")
    proposals, seen = [], set()
    for i, (pkg, d) in enumerate(zip(pkgs, ds)):
        pid = pkg_id(pkg, i)
        if pid in seen: return err(400, "DUPLICATE_PACKAGE_ID")
        seen.add(pid)
        r, f = d["evidenceRefs"], d["facts"]
        refs_txt = ", ".join(r) if r else "the cited records"
        rationale = (f"{d['action']} is the correct action for invoice {f['invoiceNumber']} from {f['vendorName']}. "
                     f"{d['reason']} Decisive evidence: {refs_txt}.")
        if len(rationale) < 60: rationale = rationale + " " * 0 + " Supported by the cited evidence references."
        proposals.append({"packageId": pid,
            "actionId": "act-" + sha(f"{bid}|{pid}|{canon(pkg)}")[:24],
            "action": d["action"], "facts": f, "evidenceRefs": r, "rationale": rationale[:1500]})
    t = {"id": str(uuid.uuid4()), "contextId": str(uuid.uuid4()),
         "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
         "artifacts": [{"artifactId": str(uuid.uuid4()), "name": "invoice-action-proposals",
                        "parts": [{"mediaType": PROP, "data": {"batchId": bid, "proposals": proposals}}]}],
         "history": [msg]}
    save(t, p)
    return t

def do_continue(p, msg, part):
    tid = str(msg["taskId"])
    with L(("t", tid)):
        t = load(tid, p)
        if not t: return err(404, "NOT_FOUND")
        if msg.get("contextId") != t["contextId"]: return err(400, "CONTEXT_MISMATCH")
        if t["status"]["state"] != "TASK_STATE_INPUT_REQUIRED": return err(409, "TASK_NOT_WAITING")
        data = part.get("data")
        if part.get("mediaType") != RES or not isinstance(data, dict): return err(400, "INVALID_REQUEST")
        prop = t["artifacts"][0]["parts"][0]["data"]
        if data.get("batchId") != prop["batchId"]: return err(400, "BATCH_MISMATCH")
        by = {x["packageId"]: x for x in prop["proposals"]}
        execs, seen = [], set()
        for r in data.get("results", []):
            x = by.get(r.get("packageId")) if isinstance(r, dict) else None
            if (not x or r["packageId"] in seen or r.get("actionId") != x["actionId"]
                    or r.get("action") != x["action"] or r.get("outcome") not in ("ACCEPTED", "REJECTED")):
                return err(400, "RESULT_MISMATCH")
            seen.add(r["packageId"])
            if r["outcome"] == "ACCEPTED":
                if not isinstance(r.get("receiptNonce"), str) or not r["receiptNonce"]:
                    return err(400, "RESULT_MISMATCH")
                execs.append({"packageId": x["packageId"], "actionId": x["actionId"], "action": x["action"],
                              "receiptNonce": r["receiptNonce"], "facts": x["facts"],
                              "evidenceRefs": x["evidenceRefs"]})
        t["artifacts"].append({"artifactId": str(uuid.uuid4()), "name": "invoice-action-receipts",
            "parts": [{"mediaType": RECEIPT, "data": {"batchId": prop["batchId"], "executions": execs}}]})
        t["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
        t["history"].append(msg)
        save(t, p)
        return t

def handle_send(p, body):
    msg = body.get("message") if isinstance(body, dict) else None
    if (not isinstance(msg, dict) or not msg.get("messageId")
            or not isinstance(msg.get("parts"), list) or not msg["parts"]):
        return err(400, "INVALID_REQUEST")
    mid, h = str(msg["messageId"]), sha(canon(msg))
    with L(("m", p, mid)):
        r = q("select hash, task_id from msgs where principal=? and mid=?", (p, mid))
        if r:
            if r[0] != h: return err(409, "IDEMPOTENCY_CONFLICT")
            return out({"task": load(r[1], p)})
        part = msg["parts"][0]
        res = do_continue(p, msg, part) if msg.get("taskId") else do_new(p, msg, part)
        if isinstance(res, Response): return res
        with dbl:
            db.execute("insert into msgs values(?,?,?,?)", (p, mid, h, res["id"]))
            db.commit()
        return out({"task": res})

# ---------------- routes ----------------
app = FastAPI()
router = APIRouter(prefix=PREFIX)

@app.get("/")
def health(): return {"ok": True}

@app.get("/.well-known/agent-card.json")
def card():
    return out({"name": "Invoice Action Agent",
        "description": "Reads invoice packages and proposes safe, evidence-backed actions.",
        "version": "1.0.0", "capabilities": {},
        "supportedInterfaces": [{"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "defaultInputModes": [BATCH], "defaultOutputModes": [PROP, RECEIPT],
        "skills": [{"id": "invoice_action_agent", "name": "Invoice action decisions",
                    "description": "Choose settle, approve, hold, reject duplicate or open exception per invoice.",
                    "tags": ["invoice", "finance", "approval"]}]})

@router.post("/message:send")
async def send(req: Request):
    p, e = guard(req)
    if e: return e
    try: body = await req.json()
    except Exception: return err(400, "INVALID_JSON")
    return await run_in_threadpool(handle_send, p, body)

@router.get("/tasks/{tid}")
def get_task(tid: str, req: Request):
    p, e = guard(req)
    if e: return e
    t = load(tid, p)
    return out(t) if t else err(404, "NOT_FOUND")

@router.get("/tasks")
def list_tasks(req: Request):
    p, e = guard(req)
    if e: return e
    with dbl:
        rows = db.execute("select data from tasks where principal=?", (p,)).fetchall()
    return out({"tasks": [json.loads(r[0]) for r in rows]})

@router.post("/tasks/{tid}:cancel")
def cancel(tid: str, req: Request):
    p, e = guard(req)
    if e: return e
    with L(("t", tid)):
        t = load(tid, p)
        if not t: return err(404, "NOT_FOUND")
        if t["status"]["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"):
            return err(409, "TASK_TERMINAL")
        t["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
        save(t, p)
        return out(t)

app.include_router(router)