"""Light tests for regex helpers."""

from core.text_clean import (
    extract_basis_points,
    extract_dates,
    extract_percentages,
    strip_boilerplate_noise,
)


def test_percentages():
    assert "2.5 %" in extract_percentages("target range 1.5 to 2.5 %")


def test_basis_points():
    assert extract_basis_points("cut by 25 basis points")


def test_dates():
    s = "January 29, 2020 statement"
    assert any("January" in d for d in extract_dates(s))


def test_strip():
    assert strip_boilerplate_noise("Hello!!!   World??") == "Hello World"
