#!/usr/bin/env python
"""Fetch a small public, open-access, GRCh38-native somatic mutation cohort from
the GDC (Genomic Data Commons) API, for demonstrating muat's native-hg38 pipeline
on real data that is NOT the iCAN WES cohort (a separate, non-public project,
documented in documentation/README_ican.md) and NOT PCAWG (hg19-only; no hg38-
lifted release exists yet, see the d5/d6 tags in experiments.json).

Downloads per-aliquot "Masked Somatic Mutation" MAF files: GDC's open-access
tier (no dbGaP/DUA needed), consensus-caller ("ensemble") somatic calls,
uniformly realigned to GRCh38. One file = one tumor sample; distinct cases are
selected per project so there is no cross-sample patient overlap by construction.

Requires internet access (this is the "fetch" step; everything downstream is
offline) -- run it on a node with network access, not a GPU-only compute node
if your cluster splits the two.

Usage
-----
    python muat/pkg_reproduce/fetch_public_hg38_cohort.py \\
        --projects TCGA-SKCM TCGA-LUSC TCGA-BRCA TCGA-COAD TCGA-PRAD TCGA-OV \\
        --n-per-project 100 --out-dir data/hg38_public_demo

Writes <out-dir>/gdc_manifest.json (file_id/case_id/project per picked sample)
and <out-dir>/gdc_maf/<file_id>.maf.gz for each entry.
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request

GDC_API = "https://api.gdc.cancer.gov"


def gdc_query(endpoint, params, timeout=30):
    url = "{}/{}?{}".format(GDC_API, endpoint, urllib.parse.urlencode(params))
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def build_manifest(projects, n_per_project):
    """Pick up to n_per_project distinct-case open MAF files per project."""
    manifest = []
    for proj in projects:
        filters = {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "data_type", "value": "Masked Somatic Mutation"}},
                {"op": "=", "content": {"field": "access", "value": "open"}},
                {"op": "=", "content": {"field": "data_format", "value": "MAF"}},
                {"op": "=", "content": {"field": "cases.project.project_id", "value": proj}},
            ],
        }
        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,file_size,cases.case_id,cases.submitter_id",
            "size": "1000",
            "format": "JSON",
        }
        hits = gdc_query("files", params)["data"]["hits"]
        seen_cases, picked = set(), []
        for h in hits:
            cases = h.get("cases", [])
            if not cases:
                continue
            case_id = cases[0]["case_id"]
            if case_id in seen_cases:
                continue  # keep at most one aliquot per case: no patient overlap
            seen_cases.add(case_id)
            picked.append({
                "project": proj,
                "file_id": h["file_id"],
                "file_name": h["file_name"],
                "case_id": case_id,
                "submitter_id": cases[0]["submitter_id"],
            })
            if len(picked) >= n_per_project:
                break
        print("{}: {} distinct cases available, {} picked".format(proj, len(seen_cases), len(picked)))
        if len(picked) < n_per_project:
            print("  WARNING: fewer cases available than requested for {}".format(proj))
        manifest.extend(picked)
    return manifest


def download_manifest(manifest, maf_dir, retries=4):
    os.makedirs(maf_dir, exist_ok=True)
    ok, failed = 0, []
    for i, m in enumerate(manifest):
        dest = os.path.join(maf_dir, m["file_id"] + ".maf.gz")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            ok += 1
            continue
        url = "{}/data/{}".format(GDC_API, m["file_id"])
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "muat-hg38-demo"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = r.read()
                with open(dest, "wb") as f:
                    f.write(data)
                ok += 1
                break
            except Exception as e:
                if attempt == retries - 1:
                    failed.append((m["file_id"], str(e)))
                else:
                    time.sleep(2)
        if (i + 1) % 50 == 0:
            print("{}/{} done, ok={}, failed={}".format(i + 1, len(manifest), ok, len(failed)))
    print("FINAL ok={} failed={}".format(ok, len(failed)))
    if failed:
        print("failed file_ids:", failed)
    return ok, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects", nargs="+", required=True,
                    help="GDC project IDs, e.g. TCGA-SKCM TCGA-LUSC TCGA-BRCA")
    ap.add_argument("--n-per-project", type=int, default=100,
                    help="max distinct cases to pick per project (default: 100)")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--manifest-only", action="store_true",
                    help="write the manifest JSON but skip downloading the MAF files")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = build_manifest(args.projects, args.n_per_project)
    manifest_path = os.path.join(args.out_dir, "gdc_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)
    print("wrote manifest: {} ({} samples)".format(manifest_path, len(manifest)))

    if not args.manifest_only:
        download_manifest(manifest, os.path.join(args.out_dir, "gdc_maf"))


if __name__ == "__main__":
    main()
