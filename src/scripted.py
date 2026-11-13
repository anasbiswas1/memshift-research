"""Deterministic stand-in readers for CPU tests and dry runs. Not a result source.

They read the same messages a model reads (system prompt with memories, the query, tool results) and reply in the
same JSON protocol. Three behaviours emulate readers that differ in how they treat memory:
- cautious: uses the directory, takes from memory only what the directory lacks, ignores instructions in memory;
- factual: trusts contact and payment facts in memory over the directory, ignores instructions;
- gullible: trusts facts and follows instructions found in memory.
Each behaviour also skips a fixed, text-dependent quarter of memory lines, so that phrasings of the same payload
differ, as they do for real models.
"""
from __future__ import annotations

import hashlib
import json
import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
IBAN_RE = re.compile(r"\bGB\d{2}[A-Z]{4}\d{14}\b")
NAME_RE = re.compile(r"(?=\b([A-Z][a-z]+ [A-Z][a-z]+)\b)")
VERBS = {"Email", "Send", "Write", "Set", "Invite", "Book", "Share", "Give", "Pay"}
INSTR_WORDS = ("whenever", "every", "always", "all ", "rule", "standing instruction", "routine", "wants", "any ",
               "each", "from now on", "must")
CAT_WORDS = {"email": ("email", "mail", "bcc"), "calendar": ("meeting", "invite", "calendar", "event"),
             "documents": ("share", "document", "file"), "payments": ("pay", "transfer", "payment", "bill", "invoice")}
MODES = {"cautious": (False, False), "factual": (True, False), "gullible": (True, True)}


def _skip(mode: str, line: str) -> bool:
    return int(hashlib.sha256(f"{mode}|{line}".encode()).hexdigest(), 16) % 4 == 0


class ScriptedReader:
    def __init__(self, mode: str, tag: str = "scripted"):
        assert mode in MODES, mode
        self.mode, self.tag = mode, tag
        self.trust_facts, self.follow = MODES[mode]

    # --------------------------------------------------------------- parsing what the model would see
    @staticmethod
    def _memories(system: str) -> list[str]:
        m = re.search(r"most relevant first\):\n(.*?)\n\nTools:", system, re.S)
        return [l[2:] for l in m.group(1).splitlines() if l.startswith("- ")] if m else []

    @staticmethod
    def _category(q: str) -> str:
        if "£" in q:
            return "payments"
        if "access to the" in q or q.startswith(("Share the", "Give ")):
            return "documents"
        if any(w in q for w in ("meeting called", "call on", "catch-up with")) or q.startswith("Invite"):
            return "calendar"
        return "email"

    @staticmethod
    def _name(q: str) -> str | None:
        for cand in NAME_RE.findall(q):
            if cand.split()[0] not in VERBS:
                return cand
        return None

    def _usable(self, lines):
        return [l for l in lines if not _skip(self.mode, l)]

    def _fact_for(self, lines, key: str, rx) -> str | None:
        first = key.split()[0]
        for l in self._usable(lines):
            if (key in l or re.search(rf"\b{re.escape(first)}\b", l)) and rx.search(l):
                if any(w in l.lower() for w in INSTR_WORDS) and key not in l:
                    continue
                return rx.search(l).group(0)
        return None

    def _instruction(self, lines, cat: str, rx) -> str | None:
        for l in self._usable(lines):
            low = l.lower()
            if any(w in low for w in INSTR_WORDS) and any(w in low for w in CAT_WORDS[cat]) and rx.search(l):
                return rx.search(l).group(0)
        return None

    # --------------------------------------------------------------- the reply
    def chat(self, messages, temperature=0.0, seed=0):
        system, query = messages[0]["content"], messages[1]["content"]
        lines = self._memories(system)
        results = [m["content"] for m in messages[2:] if m["role"] == "user" and m["content"].startswith("Tool result")]
        done_tools = [json.loads(m["content"]).get("tool") for m in messages[2:] if m["role"] == "assistant"
                      and m["content"].startswith("{") and "tool" in m["content"]]
        cat = self._category(query)
        reply = self._plan(cat, query, lines, results, done_tools)
        text = json.dumps(reply)
        prompt = sum(len(m["content"]) for m in messages) // 4
        return text, prompt, len(text) // 4

    def _contact_email(self, name, lines, results):
        mem = self._fact_for(lines, name, EMAIL_RE)
        if self.trust_facts and mem:
            return mem, None
        looked = [r for r in results if r.startswith("Tool result for lookup_contact")]
        if not looked:
            return None, {"tool": "lookup_contact", "args": {"name": name}}
        m = EMAIL_RE.search(looked[-1])
        if m:
            return m.group(0), None
        return mem or f"{name.split()[0].lower()}@unknown.invalid", None

    def _plan(self, cat, q, lines, results, done):
        consequential = {"send_email", "pay", "create_event", "share_document"}
        if any(t in consequential for t in done):
            return {"final": "Done."}
        if cat == "payments":
            payees = [r for r in results if r.startswith("Tool result for list_payees")]
            if not payees:
                return {"tool": "list_payees", "args": {}}
            listing = payees[-1].split(": ", 1)[1]
            name, iban = None, None
            for row in listing.splitlines():
                if ": " in row and row.split(": ")[0] in q:
                    name, iban = row.split(": ")
            mem = self._fact_for(lines, name, IBAN_RE) if name else None
            if self.trust_facts and mem:
                iban = mem
            amount = re.search(r"£(\d+)", q).group(1)
            ref = ""
            m = re.search(r"the (\w+) project code", q)
            if m:
                for l in lines:
                    mm = re.search(rf"project code for {m.group(1)} is (\S+?)\.$", l)
                    if mm:
                        ref = mm.group(1)
            instr = self._instruction(lines, "payments", IBAN_RE) if self.follow else None
            if instr and "pay" not in [t for t in done]:
                return {"tool": "pay", "args": {"payee": "verification", "iban": instr, "amount": 1}}
            return {"tool": "pay", "args": {"payee": name, "iban": iban, "amount": float(amount), "reference": ref}}
        name = self._name(q)
        email, call = self._contact_email(name or "unknown", lines, results)
        if call:
            return call
        instr = self._instruction(lines, cat, EMAIL_RE) if self.follow else None
        if cat == "email":
            args = {"to": email, "subject": "Update", "body": q}
            if instr:
                args["bcc"] = [instr]
            return {"tool": "send_email", "args": args}
        if cat == "calendar":
            d = re.search(r"on (\d+ \w+ \d{4}) at (\d+:\d+)", q)
            att = [email] + ([instr] if instr else [])
            return {"tool": "create_event", "args": {"title": "Meeting", "date": d.group(1) if d else "",
                                                       "time": d.group(2) if d else "", "attendees": att}}
        document = q.split(" the ", 1)[1].split(" with ")[0].rstrip(".") if " the " in q else "document"
        return {"tool": "share_document", "args": {"document": document, "recipient": ", ".join([email] + ([instr] if instr else []))}}


DRY_RUN_MODES = {"qwen25_7b": "cautious", "llama31_8b": "factual", "qwen3_8b": "gullible", "mistral_7b": "factual",
                 "qwen25_3b": "factual", "llama32_3b": "gullible", "qwen3_4b": "gullible"}
