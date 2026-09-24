"""Gmail + Google Calendar tools for miniagent, called directly via the Google REST APIs.

Credentials come from google_auth.py (run `python google_auth.py` once). Each
tool returns a plain object; call_tool() serializes it to a JSON string, so the
model always receives JSON ({"error": ...} on failure). Emails follow the shape
of gmail.users.messages.get with headers flattened to subject/from/to/date and
the body decoded; calendar events are flattened so start/end are a single
string (RFC3339 dateTime, or YYYY-MM-DD for all-day). Calendar tools act on the
primary calendar and never email attendees.
"""

import base64
import json
from datetime import date, datetime, timezone
from email.message import EmailMessage
from functools import cache

from google.auth.exceptions import RefreshError
from google.genai import types
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import google_auth

TOOLS = {}                # name -> (fn, description, json schema)
SIDE_EFFECT_TOOLS = set()  # tools that need user confirmation


def tool(description: str, properties: dict | None = None, required: list = (), side_effect: bool = False):
    def register(fn):
        schema = {"type": "object", "properties": properties or {}}
        if required:
            schema["required"] = list(required)
        TOOLS[fn.__name__] = (fn, description, schema)
        if side_effect:
            SIDE_EFFECT_TOOLS.add(fn.__name__)
        return fn
    return register


@cache
def _api(name: str, version: str):
    return build(name, version, credentials=google_auth.credentials(), cache_discovery=False)


def _gmail():
    return _api("gmail", "v1").users()


def _cal():
    return _api("calendar", "v3")


_STR = {"type": "string"}
_STRS = {"type": "array", "items": {"type": "string"}}
_INT = {"type": "integer"}

# --------------------------------------------------------------------------- #
# Gmail                                                                       #
# --------------------------------------------------------------------------- #

def _walk(part):
    yield part
    for p in part.get("parts") or []:
        yield from _walk(p)


def _decode(data: str) -> str:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", errors="replace")


def _message(msg, full: bool = False) -> dict:
    h = {x["name"].lower(): x["value"] for x in msg.get("payload", {}).get("headers", [])}
    out = {
        "id": msg.get("id"),
        "threadId": msg.get("threadId"),
        "labelIds": msg.get("labelIds", []),
        "snippet": msg.get("snippet", ""),
        **{k: h.get(k, "") for k in ("subject", "from", "to", "date")},
    }
    if full:
        text, attachments = {}, []
        for p in _walk(msg.get("payload", {})):
            body = p.get("body") or {}
            if p.get("filename"):
                attachments.append({"filename": p["filename"], "mimeType": p.get("mimeType", ""),
                                    "attachmentId": body.get("attachmentId", ""), "size": body.get("size", 0)})
            elif body.get("data"):
                text.setdefault(p.get("mimeType"), _decode(body["data"]))   # first part of each type
        out["body"] = text.get("text/plain", text.get("text/html", ""))
        out["attachments"] = attachments
    return out


def _mime(to, subject, body, cc=None, bcc=None) -> dict:
    m = EmailMessage()
    for k, v in (("To", to), ("Cc", cc), ("Bcc", bcc)):
        if v:
            m[k] = ", ".join(v)
    m["Subject"] = subject
    m.set_content(body)
    return {"raw": base64.urlsafe_b64encode(m.as_bytes()).decode()}


_MSG_ID = {"messageId": _STR}
_COMPOSE = {"to": _STRS, "subject": _STR, "body": _STR, "cc": _STRS, "bcc": _STRS}


@tool("Search emails with Gmail query syntax (e.g. 'from:x is:unread'). Returns a JSON list of message summaries.",
      {"query": _STR, "maxResults": _INT}, ["query"])
def search_emails(query: str, maxResults: int = 10):
    msgs = _gmail().messages()
    ids = msgs.list(userId="me", q=query, maxResults=maxResults).execute().get("messages", [])
    return [_message(msgs.get(userId="me", id=m["id"], format="metadata",
                              metadataHeaders=["Subject", "From", "To", "Date"]).execute()) for m in ids]


@tool("Read one email by message ID. Returns the message as JSON, including body.", _MSG_ID, ["messageId"])
def read_email(messageId: str):
    return _message(_gmail().messages().get(userId="me", id=messageId, format="full").execute(), full=True)


@tool("Send a new email.", _COMPOSE, ["to", "subject", "body"], side_effect=True)
def send_email(to: list, subject: str, body: str, cc: list | None = None, bcc: list | None = None):
    return _gmail().messages().send(userId="me", body=_mime(to, subject, body, cc, bcc)).execute()


@tool("Create a draft email.", _COMPOSE, ["to", "subject", "body"], side_effect=True)
def draft_email(to: list, subject: str, body: str, cc: list | None = None, bcc: list | None = None):
    return _gmail().drafts().create(userId="me", body={"message": _mime(to, subject, body, cc, bcc)}).execute()


@tool("Move an email to trash.", _MSG_ID, ["messageId"], side_effect=True)
def delete_email(messageId: str):
    return _message(_gmail().messages().trash(userId="me", id=messageId).execute())


@tool("Add or remove labels on an email (e.g. remove UNREAD to mark read).",
      {**_MSG_ID, "addLabelIds": _STRS, "removeLabelIds": _STRS}, ["messageId"], side_effect=True)
def modify_email(messageId: str, addLabelIds: list | None = None, removeLabelIds: list | None = None):
    body = {"addLabelIds": addLabelIds or [], "removeLabelIds": removeLabelIds or []}
    return _message(_gmail().messages().modify(userId="me", id=messageId, body=body).execute())


@tool("List all Gmail labels.")
def list_labels():
    labels = _gmail().labels().list(userId="me").execute().get("labels", [])
    return [{"id": l["id"], "name": l["name"], "type": l.get("type", "")} for l in labels]


# --------------------------------------------------------------------------- #
# Calendar                                                                    #
# --------------------------------------------------------------------------- #

CAL = "primary"


@cache
def _timezone() -> str:
    return _cal().settings().get(setting="timezone").execute().get("value", "UTC")


def _when(s: str) -> dict:
    """'YYYY-MM-DD' -> all-day; anything else is an RFC3339 dateTime in the calendar's zone."""
    try:
        date.fromisoformat(s)
        return {"date": s}
    except ValueError:
        return {"dateTime": s, "timeZone": _timezone()}


def _event(e) -> dict:
    start, end = e.get("start", {}), e.get("end", {})
    return {
        "id": e.get("id"),
        "status": e.get("status", ""),
        "summary": e.get("summary", ""),
        "description": e.get("description", ""),
        "location": e.get("location", ""),
        "start": start.get("dateTime") or start.get("date", ""),
        "end": end.get("dateTime") or end.get("date", ""),
        "organizer": e.get("organizer", {}).get("email", ""),
        "attendees": [{"email": a.get("email", ""), "responseStatus": a.get("responseStatus", "")}
                      for a in e.get("attendees", [])],
        "htmlLink": e.get("htmlLink", ""),
    }


def _event_body(summary=None, start=None, end=None, description=None, location=None, attendees=None) -> dict:
    """Only the fields that were given, in Calendar API shape."""
    body = {"summary": summary, "description": description, "location": location,
            "start": start and _when(start), "end": end and _when(end),
            "attendees": attendees and [{"email": a} for a in attendees]}
    return {k: v for k, v in body.items() if v is not None}


_TIME = {"type": "string", "description": "RFC3339 dateTime (e.g. 2026-09-25T14:00:00+02:00) or YYYY-MM-DD for all-day"}
_EVENT_ID = {"eventId": _STR}
_EVENT = {"summary": _STR, "start": _TIME, "end": _TIME, "description": _STR, "location": _STR, "attendees": _STRS}


@tool("Get the current time (UTC) and the calendar's time zone.")
def get_current_time():
    return {"now": datetime.now(timezone.utc).isoformat(timespec="seconds"), "timeZone": _timezone()}


@tool("List upcoming calendar events in a time range, optionally filtered by free-text query. timeMin defaults to now.",
      {"timeMin": _TIME, "timeMax": _TIME, "query": _STR, "maxResults": _INT})
def list_events(timeMin: str | None = None, timeMax: str | None = None, query: str | None = None, maxResults: int = 10):
    extra = {k: v for k, v in (("timeMax", timeMax), ("q", query)) if v}
    items = _cal().events().list(
        calendarId=CAL, timeMin=timeMin or datetime.now(timezone.utc).isoformat(),
        maxResults=maxResults, singleEvents=True, orderBy="startTime", **extra,
    ).execute().get("items", [])
    return [_event(e) for e in items]


@tool("Get one calendar event by ID.", _EVENT_ID, ["eventId"])
def get_event(eventId: str):
    return _event(_cal().events().get(calendarId=CAL, eventId=eventId).execute())


@tool("Create a calendar event. Attendees are added but not emailed.", _EVENT, ["summary", "start", "end"],
      side_effect=True)
def create_event(summary: str, start: str, end: str, description: str | None = None,
                 location: str | None = None, attendees: list | None = None):
    body = _event_body(summary, start, end, description, location, attendees)
    return _event(_cal().events().insert(calendarId=CAL, body=body, sendUpdates="none").execute())


@tool("Update fields of a calendar event; omitted fields are unchanged.", {**_EVENT_ID, **_EVENT}, ["eventId"],
      side_effect=True)
def update_event(eventId: str, summary: str | None = None, start: str | None = None, end: str | None = None,
                 description: str | None = None, location: str | None = None, attendees: list | None = None):
    body = _event_body(summary, start, end, description, location, attendees)
    return _event(_cal().events().patch(calendarId=CAL, eventId=eventId, body=body, sendUpdates="none").execute())


@tool("Delete a calendar event.", _EVENT_ID, ["eventId"], side_effect=True)
def delete_event(eventId: str):
    _cal().events().delete(calendarId=CAL, eventId=eventId, sendUpdates="none").execute()
    return {"id": eventId, "deleted": True}


# --------------------------------------------------------------------------- #
# dispatch                                                                    #
# --------------------------------------------------------------------------- #

def gemini_tool() -> types.Tool:
    return types.Tool(function_declarations=[
        types.FunctionDeclaration(name=name, description=desc, parameters_json_schema=schema)
        for name, (_, desc, schema) in TOOLS.items()
    ])


def call_tool(name: str, args: dict) -> str:
    """Run a tool synchronously; always returns a JSON string."""
    if name not in TOOLS:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        return json.dumps(TOOLS[name][0](**args), indent=2, ensure_ascii=False)
    except HttpError as e:
        err = f"Google API {e.status_code}: {e.reason}"
    except RefreshError as e:
        err = f"auth failed, run `python google_auth.py`: {e}"
    except Exception as e:
        err = repr(e)
    return json.dumps({"error": err}, ensure_ascii=False)
