from src.preprocess import lemmas, merge_split_words, normalize


def test_lowercase_and_yo():
    assert normalize("ЁМКОСТЬ") == "емкость"


def test_masked_serials_removed():
    toks = lemmas("ЗАВ. НОМЕРА С 5XXXXX7 ПО 5XXXXX5 ТРУБКИ")
    assert "5xxxxx7" not in toks and "трубка" in toks


def test_split_words_merged():
    assert merge_split_words(["т", "вердотельных"]) == ["твердотельных"]
    assert merge_split_words(["каро", "тажа"]) == ["каротажа"]
    # два нормальных слова не склеиваем
    assert merge_split_words(["насос", "струйный"]) == ["насос", "струйный"]


def test_lemmas_are_normal_forms():
    assert lemmas("НАКОПИТЕЛЕЙ ТВЕРДОТЕЛЬНЫХ") == ["накопитель", "твердотельный"]
