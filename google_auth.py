"""Shared Google OAuth credentials

One-time setup (opens a consent page; grants Gmail + Calendar):
    python google_auth.py

This writes ~/.gmail-mcp/google-token.json, which every tool call then uses.
"""

from pathlib import Path

from google.oauth2.credentials import Credentials

CRED_DIR = Path.home() / ".gmail-mcp"
CLIENT_FILE = CRED_DIR / "gcp-oauth.keys.json"
TOKEN_FILE = CRED_DIR / "google-token.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.settings.basic",
    "https://www.googleapis.com/auth/calendar",
]

_creds = None


def credentials() -> Credentials:
    global _creds
    if _creds is None:
        if not TOKEN_FILE.exists():
            raise FileNotFoundError(f"{TOKEN_FILE} missing; run `python google_auth.py` first")
        _creds = Credentials.from_authorized_user_file(str(TOKEN_FILE))
    return _creds


def authorize() -> None:
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_FILE), SCOPES)
    # WSL usually can't launch a browser; print the URL and open it on the host.
    creds = flow.run_local_server(port=0, open_browser=False, prompt="consent")
    TOKEN_FILE.write_text(creds.to_json())
    print(f"saved {TOKEN_FILE}\nscopes: {' '.join(creds.scopes or [])}")


if __name__ == "__main__":
    authorize()
