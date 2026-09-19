from src.corpus import build_docs, find_refs, split_prefix


def test_split_prefix():
    assert split_prefix("Раздел 1, 6.1.2.4.1. криогенные охладители") == ("6.1.2.4.1", "криогенные охладители")
    assert split_prefix("Угловые измерительные приборы") == (None, "Угловые измерительные приборы")


def test_find_refs_lists():
    assert find_refs("указанные в пункте 6.1.6.1, 6.1.6.2 или 6.1.6.3. прим") == ["6.1.6.1", "6.1.6.2", "6.1.6.3"]
    assert find_refs('в подпункте "а" пункта 6.1.4.4.2') == ["6.1.4.4.2"]


def test_refs_resolved_within_decree_only():
    regs = [
        {"regulation_id": "A", "decree_number": "1661", "npa": "Раздел 1, 1.1. лазеры мощные"},
        {"regulation_id": "B", "decree_number": "1661", "npa": "Раздел 1, 1.2. зеркала для лазеров, указанных в пункте 1.1"},
        {"regulation_id": "C", "decree_number": "36", "npa": "детали для изделий, указанных в пункте 1.1"},
    ]
    docs = {d.regulation_id: d for d in build_docs(regs)}
    assert docs["B"].ref_texts == ["лазеры мощные"]
    assert docs["C"].refs == ["1.1"] and docs["C"].ref_texts == []
    assert "Связанные позиции" in docs["B"].text_for_model()
