# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pytest

import consenrich.matching as matching

NAME_RE = re.compile(
    r"^consenrichPeak\|i=(?P<i>\d+)\|gap=(?P<gap>\d+)bp\|ct=(?P<ct>\d+)\|qRange=(?P<qmin>\d+\.\d{3})_(?P<qmax>\d+\.\d{3})$"
)

def write_np(path, rows):
    path.write_text("\n".join(rows) + "\n")
    return str(path)

def parse_name(name):
    m = NAME_RE.match(name)
    assert m, f"bad merged name field: {name}"
    return {
        "i": int(m["i"]),
        "gap": int(m["gap"]),
        "ct": int(m["ct"]),
        "qmin": float(m["qmin"]),
        "qmax": float(m["qmax"]),
    }

def read_lines(path):
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]

def fields(line):
    f = line.split("\t")
    assert len(f) == 10, f"narrowPeak requires 10 fields got {len(f)}"
    chrom, s, e, name, score, strand, sig, pHM, qHM, peak = f
    return {
        "chrom": chrom,
        "start": int(s),
        "end": int(e),
        "name": name,
        "score": int(score),
        "strand": strand,
        "sig": float(sig),
        "pHM": float(pHM),
        "qHM": float(qHM),
        "peak": int(peak),
    }

def test_merge_two_within_gap(tmp_path):
    p = tmp_path / "a.narrowPeak"
    rows = [
        "chr1\t1000\t1100\tx\t500\t.\t5.0\t2.000\t1.500\t50",
        "chr1\t1120\t1200\ty\t700\t.\t6.0\t3.000\t2.000\t40",
    ]
    path = write_np(p, rows)
    out = matching.mergeMatches(path, mergeGapBP=75)
    assert out and os.path.isfile(out)
    ls = read_lines(out)
    assert len(ls) == 1
    f = fields(ls[0])
    meta = parse_name(f["name"])
    assert meta["gap"] == 75
    assert meta["ct"] == 2
    # merged span should cover both
    assert f["start"] == 1000
    assert f["end"] >= 1200
    # score should be clamped to [0, 1000]
    assert 0 <= f["score"] <= 1000
    # qHM within [min q, max q] after converting log10 form
    qmin = 10 ** (-meta["qmax"])
    qmax = 10 ** (-meta["qmin"])
    qhm = 10 ** (-f["qHM"])
    assert qmin <= qhm <= qmax

def test_do_not_merge_across_chromosomes(tmp_path):
    p = tmp_path / "b.narrowPeak"
    rows = [
        "chr1\t1000\t1050\tx\t500\t.\t5.0\t2.000\t1.500\t25",
        "chr2\t1010\t1060\ty\t500\t.\t5.0\t2.000\t1.500\t25",
    ]
    out = matching.mergeMatches(write_np(p, rows), mergeGapBP=1000)
    ls = read_lines(out)
    # should remain separate
    assert len(ls) == 2
    chroms = [fields(x)["chrom"] for x in ls]
    assert set(chroms) == {"chr1", "chr2"}

def test_exact_gap_merges(tmp_path):
    p = tmp_path / "c.narrowPeak"
    # first ends at 1100, second starts at 1100 + 75 exact gap
    rows = [
        "chr3\t1000\t1100\ta\t100\t.\t1.0\t1.000\t1.000\t50",
        "chr3\t1175\t1225\tb\t200\t.\t2.0\t1.000\t1.000\t25",
    ]
    out = matching.mergeMatches(write_np(p, rows), mergeGapBP=75)
    ls = read_lines(out)
    assert len(ls) == 1
    meta = parse_name(fields(ls[0])["name"])
    assert meta["ct"] == 2

def test_gap_plus_one_does_not_merge(tmp_path):
    p = tmp_path / "d.narrowPeak"
    # gap is 76 so should not merge with gap=75
    rows = [
        "chr4\t1000\t1100\ta\t100\t.\t1.0\t1.000\t1.000\t50",
        "chr4\t1176\t1225\tb\t200\t.\t2.0\t1.000\t1.000\t25",
    ]
    out = matching.mergeMatches(write_np(p, rows), mergeGapBP=75)
    ls = read_lines(out)
    assert len(ls) == 2

def test_unsorted_input_is_ok(tmp_path):
    p = tmp_path / "e.narrowPeak"
    # intentionally out of order
    rows = [
        "chr5\t2000\t2050\tb\t900\t.\t3.0\t2.000\t2.000\t25",
        "chr5\t1000\t1100\ta\t900\t.\t3.0\t2.000\t2.000\t50",
        "chr5\t1110\t1150\tc\t900\t.\t3.0\t2.000\t2.000\t20",
    ]
    out = matching.mergeMatches(write_np(p, rows), mergeGapBP=75)
    ls = read_lines(out)
    # first two should merge into one, the 2000..2050 should remain separate
    assert len(ls) == 2
    spans = [(fields(x)["start"], fields(x)["end"]) for x in ls]
    spans.sort()
    assert spans[0][0] == 1000 and spans[0][1] >= 1150
    assert spans[1][0] == 2000

def test_score_clamping_and_pointsource_fallback(tmp_path):
    p = tmp_path / "f.narrowPeak"
    # very large scores should clamp to 1000
    # negative or small ranges exercise pointSource fallback path
    rows = [
        "chr6\t1000\t1010\ta\t50000\t.\t2.0\t2.000\t1.000\t-1",
        "chr6\t1020\t1025\tb\t50000\t.\t2.0\t2.000\t1.000\t-1",
    ]
    out = matching.mergeMatches(write_np(p, rows), mergeGapBP=50)
    f = fields(read_lines(out)[0])
    assert f["score"] == 1000
    # point field exists and is inside span
    assert 0 <= f["peak"] <= (f["end"] - f["start"])

@pytest.mark.parametrize("pvals", [
    [2.0, 3.0],            # typical finite log10 p
    [10.0, 10.0],          # large values saturate at MAX_NEGLOGP internally
])
def test_pHM_monotone_bounds(tmp_path, pvals):
    p = tmp_path / "g.narrowPeak"
    rows = [
        f"chr7\t1000\t1100\tx\t100\t.\t1.0\t{pvals[0]:.3f}\t2.000\t50",
        f"chr7\t1120\t1200\ty\t100\t.\t1.0\t{pvals[1]:.3f}\t2.000\t50",
    ]
    out = matching.mergeMatches(write_np(p, rows), mergeGapBP=100)
    f = fields(read_lines(out)[0])
    # pHM is a harmonic meanish aggregate on log10 scale and should be <= max component
    assert f["pHM"] <= max(pvals) + 1e-6
