"""Authentication helpers for connecting to remote agent-plane servers.

Provides a single entry point :func:`authenticate` that detects the
server type and returns HTTP headers for authenticated requests.

Currently supports:
- Databricks Apps (``*.databricksapps.com``) — browser-based OAuth
- Plain HTTP servers — no auth needed
"""

from __future__ import annotations

import sys

import httpx


def authenticate(server_url: str) -> dict[str, str]:
    """
    Authenticate to a remote agent-plane server.

    Detects the server type from the URL and performs the
    appropriate auth flow. Returns headers to include in all
    HTTP requests to the server.

    :param server_url: The server URL, e.g.
        ``"https://my-app.databricksapps.com"`` or
        ``"http://localhost:8000"``.
    :returns: HTTP headers dict (empty if no auth needed).
    """
    if "databricksapps.com" not in server_url:
        return {}

    # Check if auth is actually required
    try:
        resp = httpx.get(
            f"{server_url}/health",
            timeout=10,
            follow_redirects=False,
        )
        if resp.status_code == 200:
            return {}
    except Exception:
        pass

    return _databricks_oauth(server_url)


def _databricks_oauth(server_url: str) -> dict[str, str]:
    """
    Authenticate to a Databricks App via browser-based OAuth.

    Uses the ``databricks-cli`` registered OAuth client which
    supports U2M flows with PKCE + localhost callback. Opens the
    user's browser for consent, exchanges the authorization code
    for an access token, and returns it as a Bearer header.

    Requires ``DATABRICKS_HOST`` env var set to the workspace URL
    and ``databricks-sdk`` installed.

    :param server_url: The Databricks App URL, e.g.
        ``"https://my-app.databricksapps.com"``.
    :returns: Headers with ``Authorization: Bearer <token>``.
    """
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.oauth import OAuthClient, OidcEndpoints
    except ImportError:
        print("databricks-sdk is required to connect to Databricks Apps")
        print("  pip install databricks-sdk")
        sys.exit(1)

    wc = WorkspaceClient()
    host = wc.config.host.rstrip("/")

    # Discover OIDC endpoints from the workspace
    oidc_resp = httpx.get(
        f"{host}/oidc/.well-known/openid-configuration",
        timeout=10,
    )
    oidc_resp.raise_for_status()
    oidc = oidc_resp.json()

    # Use the Databricks CLI's registered OAuth client — it
    # supports localhost redirect + PKCE for U2M flows.
    oauth = OAuthClient(
        oidc_endpoints=OidcEndpoints(
            authorization_endpoint=oidc["authorization_endpoint"],
            token_endpoint=oidc["token_endpoint"],
        ),
        client_id="databricks-cli",
        redirect_url="http://localhost:8020",
        scopes=["all-apis"],
    )

    print("Opening browser for Databricks authentication...")
    consent = oauth.initiate_consent()
    creds = consent.launch_external_browser()
    token = creds.token()

    headers = {"Authorization": f"Bearer {token.access_token}"}

    # Verify the token works
    resp = httpx.get(
        f"{server_url}/health",
        headers=headers,
        timeout=10,
        follow_redirects=False,
    )
    if resp.status_code != 200:
        print(f"Authentication failed — server returned {resp.status_code}")
        sys.exit(1)

    print("Authenticated successfully!\n")
    return headers
