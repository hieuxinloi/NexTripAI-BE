from __future__ import annotations

import argparse

from src.config import Settings


PRIVILEGED_ROLES = ("user", "support", "admin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign a server-verified NexTripAI role as a Firebase custom claim.",
    )
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--uid", help="Firebase Authentication UID")
    identity.add_argument("--email", help="Firebase Authentication email")
    parser.add_argument("role", choices=PRIVILEGED_ROLES)
    parser.add_argument(
        "--revoke-sessions",
        action="store_true",
        help="Revoke refresh tokens so the account must authenticate again.",
    )
    return parser.parse_args()


def initialize_firebase(app_settings: Settings) -> None:
    import firebase_admin
    from firebase_admin import credentials

    try:
        firebase_admin.get_app()
        return
    except ValueError:
        pass

    credential = None
    if app_settings.firestore_credentials_path:
        credential = credentials.Certificate(app_settings.firestore_credentials_path)
    options = (
        {"projectId": app_settings.firebase_project_id}
        if app_settings.firebase_project_id
        else None
    )
    firebase_admin.initialize_app(credential, options=options)


def sanitized_claims(current: dict[str, object] | None, role: str) -> dict[str, object]:
    claims = dict(current or {})
    claims.pop("admin", None)
    existing_roles = claims.pop("roles", None)
    if isinstance(existing_roles, (list, tuple, set)):
        retained = [
            value
            for value in existing_roles
            if str(value).strip().lower() not in PRIVILEGED_ROLES
        ]
        if retained:
            claims["roles"] = retained
    claims["role"] = role
    return claims


def main() -> None:
    from firebase_admin import auth

    args = parse_args()
    initialize_firebase(Settings())
    user = auth.get_user(args.uid) if args.uid else auth.get_user_by_email(args.email)
    claims = sanitized_claims(user.custom_claims, args.role)
    auth.set_custom_user_claims(user.uid, claims)
    if args.revoke_sessions:
        auth.revoke_refresh_tokens(user.uid)
    print(
        f"Updated Firebase role: uid={user.uid} email={user.email or '-'} "
        f"role={args.role}. Sign out and back in to refresh the ID token."
    )


if __name__ == "__main__":
    main()
