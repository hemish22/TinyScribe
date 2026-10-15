import pytest

from tinyscribe.normalize import normalize


@pytest.mark.parametrize("a, b", [
    ("०९/०५/२०००", "09/05/2000"),            # Devanagari digits
    ("৮৮১৯৭৩০৯০৯", "8819730909"),            # Bengali digits
    ("௨௦௦௫", "2005"),                        # Tamil digits
    ("9/5/2000", "2000-05-09"),               # date forms agree
    ("09-05-2000", "2000.05.09"),
    ("  Divya   Sharma ", "divya sharma"),    # whitespace + case
    ("Bhopal - 462001", "Bhopal-462001"),
    ("42 / B", "42/B"),
    ("Station Road ,Jaipur", "Station Road, Jaipur"),
    ("Clerk.", "Clerk"),
    ("कु‍मार", "कुमार"),                 # zero-width joiner dropped
    ("क़", "क़"),                       # NFC: decomposed nukta == precomposed
])
def test_equal(a, b):
    assert normalize(a) == normalize(b)


@pytest.mark.parametrize("a, b", [
    ("मीना", "मीनी"),
    ("2000", "2001"),
    ("TEST-0001", "TEST-0010"),
])
def test_different(a, b):
    assert normalize(a) != normalize(b)


def test_blank():
    assert normalize(None) == normalize("") == normalize("   ") == ""
