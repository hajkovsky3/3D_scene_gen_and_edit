from scenepipe.core.workspace import BUILD_PHASES, EDIT_PHASES, PHASES, Workspace


def test_phase_dirs(tmp_path):
    ws = Workspace.open(tmp_path, "demo")
    assert ws.phase_dir("01_sfm") == tmp_path / "demo" / "01_sfm"
    assert ws.phase_dir("11_editing", "e001") == tmp_path / "demo" / "edits" / "e001" / "11_editing"
    assert len(PHASES) == 13 and len(BUILD_PHASES) == 8 and len(EDIT_PHASES) == 4
