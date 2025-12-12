"""
Legacy entrypoint for the original CT_to_3D pipeline.

For now this is just a placeholder; later we will:
- move/refactor the original CT_to_3D.py logic into ct23d.core
- expose a compatibility main() in ct23d.core.ct_compat
- call that from here
"""

def main() -> None:
    raise SystemExit(
        "Legacy pipeline not yet wired. "
        "Once refactored, this script will call ct23d.core.ct_compat.main()."
    )


if __name__ == "__main__":
    main()
