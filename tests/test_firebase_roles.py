from scripts.set_firebase_role import sanitized_claims


def test_role_update_removes_older_privileged_claim_shapes() -> None:
    claims = sanitized_claims(
        {
            "admin": True,
            "role": "admin",
            "roles": ["admin", "support", "billing"],
            "feature_flag": "enabled",
        },
        "user",
    )

    assert claims == {
        "role": "user",
        "roles": ["billing"],
        "feature_flag": "enabled",
    }
