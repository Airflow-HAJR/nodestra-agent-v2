"""
Supabase account verification.

Accounts are optional: anyone can talk to the agent as a guest. Signing in
(Google, via Supabase Auth) is what makes preferences durable — the account's
Supabase user id becomes the stable memory key, so the same person is
recognised on their next visit, on any device.

The browser holds a Supabase access token (a JWT). This module turns that token
into a verified identity by asking Supabase's auth API who it belongs to. We do
NOT decode the JWT locally: verifying the signature needs the project's JWT
secret, which this server has no business holding, and a revoked/expired token
must stop working immediately rather than at its own `exp`.

Results are cached briefly so a chatty client doesn't hit the auth API on every
request; the cache is keyed by the token itself, so signing out (which mints no
new token) can never resurrect an old identity for longer than the TTL.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from agent.config import SUPABASE_KEY, SUPABASE_URL

logger = logging.getLogger(__name__)

# How long a verified token is trusted without re-asking Supabase. Short enough
# that a signed-out session stops being honoured almost immediately, long enough
# that the account sheet's few calls cost one round trip between them.
_CACHE_TTL_SECONDS = 60.0
_NEGATIVE_TTL_SECONDS = 15.0  # don't hammer the auth API with a token it already rejected

_cache: dict[str, tuple[Optional["Account"], float]] = {}


@dataclass(frozen=True)
class Account:
    """A verified signed-in user."""

    id: str  # Supabase auth user uuid
    email: str | None
    name: str | None
    avatar_url: str | None

    @property
    def user_id(self) -> str:
        """The key memories are stored under.

        Namespaced so it can never collide with the random per-device guest id
        the browser generates for anonymous sessions, or with a phone number
        from the Twilio side.
        """
        return f"sub:{self.id}"

    @property
    def first_name(self) -> str | None:
        if not self.name:
            return None
        return self.name.strip().split(" ")[0] or None


def auth_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _account_from_payload(data: dict) -> Optional[Account]:
    user_id = data.get("id")
    if not user_id:
        return None
    meta = data.get("user_metadata") or {}
    name = meta.get("full_name") or meta.get("name") or meta.get("preferred_username")
    return Account(
        id=str(user_id),
        email=data.get("email") or meta.get("email"),
        name=name,
        avatar_url=meta.get("avatar_url") or meta.get("picture"),
    )


def _cached(token: str) -> tuple[bool, Optional[Account]]:
    """(hit, account). A cached rejection is a hit whose account is None."""
    entry = _cache.get(token)
    if entry and time.monotonic() < entry[1]:
        return True, entry[0]
    if entry:
        _cache.pop(token, None)
    return False, None


def _store(token: str, account: Optional[Account]) -> None:
    ttl = _CACHE_TTL_SECONDS if account else _NEGATIVE_TTL_SECONDS
    _cache[token] = (account, time.monotonic() + ttl)
    # Unbounded growth isn't a real risk at this scale, but a runaway client
    # shouldn't be able to pin memory either.
    if len(_cache) > 1000:
        now = time.monotonic()
        for k, (_, exp) in list(_cache.items()):
            if exp <= now:
                _cache.pop(k, None)


async def verify_access_token(token: str | None) -> Optional[Account]:
    """Resolve a Supabase access token to an Account, or None if it isn't valid.

    Never raises: an unreachable auth API means "not signed in", which degrades
    to the guest experience rather than breaking the conversation.
    """
    token = (token or "").strip()
    if not token or not auth_configured():
        return None

    hit, cached = _cached(token)
    if hit:
        return cached

    try:
        async with httpx.AsyncClient(timeout=6) as client:
            resp = await client.get(
                f"{SUPABASE_URL.rstrip('/')}/auth/v1/user",
                headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_KEY},
            )
        if resp.status_code != 200:
            _store(token, None)
            return None
        account = _account_from_payload(resp.json() or {})
    except Exception as exc:
        # Transport failure — don't cache it, the next attempt may well work.
        logger.warning(f"Supabase token verification failed: {exc}")
        return None

    _store(token, account)
    return account


def bearer_token(authorization: str | None) -> str | None:
    """Pull the token out of an `Authorization: Bearer <jwt>` header."""
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None
