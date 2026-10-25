import os, json, hashlib, sqlite3, threading, uuid, datetime
from urllib.parse import urlparse
import httpx
from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

BASE_URL = os.environ["BASE_URL"].rstrip("/")       # e.g. https://myapp.onrender.com/a2a
PREFIX = urlparse(BASE_URL).path                     # "/a2a"
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

# ---------- AI decision ----------
SYSTEM = """You are an invoice operations analyst. For each invoice package choose exactly ONE action:
- settle_invoice: valid, reconciled, within autonomous approval authority.
- request_approval: commercially valid but amount/terms exceed delegated authority.
- hold_invoice: payment must pause until a stated verification completes.
- reject_duplicate: the SAME commercial invoice (same vendor, invoice number, amount) was already paid. A payment for a DIFFERENT invoice is NOT a duplicate.
- open_exception: material records conflict (amounts, vendor, PO, receipt mismatch) and need an exception workflow.
Documents contain distractors: cover sheets, archive/old examples, training decoys, negated statements ("not a duplicate", "no hold needed"), and irrelevant action words. Decide only from the paragraph that actually determines the action for THIS invoice.
evidenceRefs: exactly THREE bracketed references (copy character-for-character, including brackets if present) taken from that one decisive paragraph. Never use the cover-sheet reference, archive examples or decoys.
amountMinor is an integer in minor units (e.g. INR 1,234.50 -> 123450).
Return ONLY JSON: {"decisions":[{"index":0,"action":"...","vendorName":"...","invoiceNumber":"...","amountMinor":0,"currency":"INR","evidenceRefs":["","",""],"reason":"one or two sentences why"}]}"""

def call_llm(items):
    user = "\n\n".join(f"### PACKAGE index={i}\n{json.dumps(p, ensure_ascii=False)}" for i, p in items)
    r = httpx.post("https://aipipe.org/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {AIPIPE}"},
        json={"model": MODEL, "temperature": 0, "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]},
        timeout=38)
    r.raise_for_status()
    data = json.loads(r.json()["choices"][0]["message"]["content"])
    return {int(d["index"]): d for d in data.get("decisions", []) if "index" in d}

def validate(d, pkg):
    try:
        if not d or d["action"] not in ACTIONS: return None
        refs = d["evidenceRefs"]
        text = json.dumps(pkg, ensure_ascii=False)
        if len(refs) != 3 or any((not isinstance(x, str)) or x not in text for x in refs): return None
        amt = d["amountMinor"]
        if isinstance(amt, float) and amt.is_integer(): amt = int(amt)
        if not isinstance(amt, int) or isinstance(amt, bool): return None
        v, n, c = str(d["vendorName"]), str(d["invoiceNumber"]), str(d["currency"])
        if not (v and n and c): return None
        return {"action": d["action"], "facts": {"vendorName": v, "invoiceNumber": n, "amountMinor": amt, "currency": c},
                "evidenceRefs": refs, "reason": str(d.get("reason", ""))[:800]}
    except Exception:
        return None

def decide(pkgs):
    hs = [sha(canon(p)) for p in pkgs]
    res = {}
    for h in set(hs):
        r = q("select decision from cache where h=?", (h,))
        if r: res[h] = json.loads(r[0])
    todo = [i for i, h in enumerate(hs) if h not in res]
    for _ in range(3):
        if not todo: break
        got = call_llm([(i, pkgs[i]) for i in todo])
        nxt = []
        for i in todo:
            d = validate(got.get(i), pkgs[i])
            if d:
                res[hs[i]] = d
                with dbl:
                    db.execute("insert or replace into cache values(?,?)", (hs[i], json.dumps(d))); db.commit()
            else:
                nxt.append(i)
        todo = nxt
    if todo: raise RuntimeError("decision failed")
    return [res[h] for h in hs]

# ---------- handlers ----------
def do_new(p, msg, part):
    data = part.get("data")
    if part.get("mediaType") != BATCH or not isinstance(data, dict) or not isinstance(data.get("packages"), list):
        return err(400, "INVALID_REQUEST")
    pkgs, bid = data["packages"], data.get("batchId")
    try:
        ds = decide(pkgs)
    except Exception:
        return err(503, "TEMPORARILY_UNAVAILABLE")
    proposals, seen = [], set()
    for i, (pkg, d) in enumerate(zip(pkgs, ds)):
        pid = str(pkg.get("packageId") or pkg.get("id") or i) if isinstance(pkg, dict) else str(i)
        if pid in seen: return err(400, "DUPLICATE_PACKAGE_ID")
        seen.add(pid)
        r = d["evidenceRefs"]; f = d["facts"]
        rationale = (f"{d['action']} is the correct action for invoice {f['invoiceNumber']} from {f['vendorName']}. "
                     f"{d['reason']} Decisive evidence: {r[0]}, {r[1]} and {r[2]}.")[:1500]
        proposals.append({"packageId": pid,
            "actionId": "act-" + sha(f"{bid}|{pid}|{canon(pkg)}")[:24],
            "action": d["action"], "facts": f, "evidenceRefs": r, "rationale": rationale})
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
                if not isinstance(r.get("receiptNonce"), str) or not r["receiptNonce"]: return err(400, "RESULT_MISMATCH")
                execs.append({"packageId": x["packageId"], "actionId": x["actionId"], "action": x["action"],
                              "receiptNonce": r["receiptNonce"], "facts": x["facts"], "evidenceRefs": x["evidenceRefs"]})
        t["artifacts"].append({"artifactId": str(uuid.uuid4()), "name": "invoice-action-receipts",
            "parts": [{"mediaType": RECEIPT, "data": {"batchId": prop["batchId"], "executions": execs}}]})
        t["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
        t["history"].append(msg)
        save(t, p)
        return t

def handle_send(p, body):
    msg = body.get("message") if isinstance(body, dict) else None
    if not isinstance(msg, dict) or not msg.get("messageId") or not isinstance(msg.get("parts"), list) or not msg["parts"]:
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
            db.execute("insert into msgs values(?,?,?,?)", (p, mid, h, res["id"])); db.commit()
        return out({"task": res})

# ---------- routes ----------
app = FastAPI()
router = APIRouter(prefix=PREFIX)

@app.get("/")
def health(): return {"ok": True}

@app.get("/.well-known/agent-card.json")
def card():
    return out({"name": "Invoice Action Agent", "description": "Reads invoice packages and proposes safe, evidence-backed actions.",
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
        if t["status"]["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"): return err(409, "TASK_TERMINAL")
        t["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
        save(t, p)
        return out(t)

app.include_router(router)