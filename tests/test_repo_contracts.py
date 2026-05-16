from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_MARKERS = (
    "https://cdn.",
    "https://cdnjs.cloudflare.com",
    "https://cdn.jsdelivr.net",
    "https://unpkg.com",
    "https://fonts.googleapis.com",
    "https://fonts.gstatic.com",
)


def test_release_contract_files_exist():
    assert (PROJECT_ROOT / "E156-PROTOCOL.md").exists()
    assert (
        (PROJECT_ROOT / "docs" / "index.html").exists()
        or (PROJECT_ROOT / "docs" / "e156_micro_paper.txt").exists()
    )


def test_dashboard_is_offline_first():
    dashboard_path = PROJECT_ROOT / "docs" / "index.html"
    if not dashboard_path.exists():
        return
    html = dashboard_path.read_text(encoding="utf-8")
    assert not any(marker in html for marker in OFFLINE_MARKERS)


def test_protocol_has_dashboard_reference():
    protocol = (PROJECT_ROOT / "E156-PROTOCOL.md").read_text(encoding="utf-8")
    assert "**Dashboard**:" in protocol
