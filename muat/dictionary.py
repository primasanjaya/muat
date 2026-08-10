"""Build muat's three token dictionaries from a directory of annotated files.

Needed whenever the reference genome changes. `dictChpos.tsv` maps `<chrom>_<Mb bin>` to
a token id, so it is genome-build specific: the shipped one is hg19-derived, and using it
for native GRCh38 data silently looks hg38 coordinates up in hg19 bins.

What can and cannot be rebuilt from annotated files:

  pos    YES -- derived from chrom + pos, which annotated files carry.
  ges    YES -- derived from genic/exonic/strand, which annotated files carry.
  motif  PARTLY -- the seq strings are present, but their `mut_type` label is NOT.
         Annotated files have columns chrom/pos/ref/alt/sample/seq/genic/exonic/strand
         and no mut_type, and the label cannot be reconstructed: it is not a function of
         allele length (in PCAWG-open every row is single-base yet the shipped dictionary
         labels most of those motifs MNV) and not a function of the seq separator either
         (270 separator patterns map to more than one mut_type). So mut_type labels are
         INHERITED from an existing dictionary, which is sound because sequence motifs do
         not depend on the genome build -- only coordinates do.

Deliberately NOT a wrapper around preprocessing.create_dictionary(): that function
rewrites every input file in place (adding chrompos/ges columns) and serialises unordered
sets to a single JSON. Token ids assigned from unordered sets are not reproducible between
runs, which would make the resulting model unreproducible. Everything here is read-only
and fully sorted.
"""

import os
import glob as _glob

import pandas as pd

from .util import (resolve_path, ensure_dirpath, ANNOTATED_GLOBS,
                   ANNOTATED_SUFFIXES_ACCEPTED)
from ._resources import pkg_path

# Token ids must be CONTIGUOUS and grouped in this order. model.py sizes the motif
# embeddings from per-class row counts (vocabSNV + vocabMNV + vocabindel + vocabSVMEI +
# vocabNormal), so a different grouping silently mis-slices the vocabulary. Verified
# against the shipped dictionary: SNV 1..96, MNV 97..2266, indel 2267..3426,
# MEI 3427..3459, SV 3460..3692, Neg 3693..4830.
MUT_TYPE_ORDER = ['SNV', 'MNV', 'indel', 'MEI', 'SV', 'Neg']

# model.py:88 additionally *requires* all six classes to be present, but spells the last
# one 'Normal' where the shipped dictionary uses 'Neg' -- a pre-existing mismatch, noted
# so a rebuilt dictionary is not blamed for it.
REQUIRED_MUT_TYPES = set(MUT_TYPE_ORDER)

# Karyotype order, so dictChpos reads 1_0, 1_1, ... X_*, Y_* like the shipped file.
CHROM_ORDER = [str(i) for i in range(1, 23)] + ['X', 'Y']

DEFAULT_POS_BIN_SIZE = 1000000
# None -> both the current (.annotate.tsv*) and legacy (.gc.genic.exonic.cs.tsv*) namings,
# per util.ANNOTATED_GLOBS. A string here overrides with a single explicit glob.
ANNOTATED_GLOB = None


def default_motif_dictionary():
    """The shipped hg19 dictMutation.tsv, used as the source of mut_type labels."""
    return os.path.join(ensure_dirpath(pkg_path('extfile')), 'dictMutation.tsv')


def load_mut_type_labels(path=None):
    """seq -> mut_type from an existing dictMutation.tsv."""
    path = resolve_path(path) if path else default_motif_dictionary()
    if not os.path.exists(path):
        raise FileNotFoundError(f"motif dictionary not found: {path}")
    df = pd.read_csv(path, sep='\t')
    for col in ('seq', 'mut_type'):
        if col not in df.columns:
            raise ValueError(f"{path} has no '{col}' column; expected a dictMutation.tsv "
                             f"with seq/triplettoken/mut_type.")
    return dict(zip(df['seq'].astype(str), df['mut_type'].astype(str))), path


def _chrom_sort_key(chrom):
    try:
        return (0, CHROM_ORDER.index(str(chrom)))
    except ValueError:
        return (1, str(chrom))


# --- motif labelling -------------------------------------------------------------------
#
# mut_type cannot be recovered from a motif string. Measured against the shipped
# dictionary, the purest structural bucket is only 96.7% and the largest is 47.1%, because
# `Neg` is a SAMPLING category (contexts drawn from unmutated positions, which the
# sweepline still fills with neighbouring mutations) and so has no structural signature at
# all, and because a motif encodes neighbours as well as the focal event -- so "contains an
# indel code" does not mean the focal variant was an indel.
#
# What IS available at build time is ref/alt, from the annotated files. Allele lengths give
# a deterministic SNV/MNV/indel call. Note this answers a DIFFERENT question than the
# shipped labels: "what variant is this" rather than "what kind of context is this motif".
# The two disagree often -- in PCAWG-open every row is single-base, yet the shipped
# dictionary calls 512 of those motifs MNV. Hence the modes below, and the provenance file.
MOTIF_LABEL_MODES = ('inherit', 'hybrid', 'refalt')


def classify_from_alleles(ref, alt):
    """SNV / MNV / indel from allele lengths. Cannot yield SV, MEI or Neg."""
    ref, alt = str(ref), str(alt)
    if ',' in alt:                      # multi-allelic: not a single event
        return None
    if len(ref) == len(alt):
        return 'SNV' if len(ref) == 1 else 'MNV'
    return 'indel'


def find_annotated_files(data_dir, pattern=ANNOTATED_GLOB):
    """Annotated inputs in data_dir, excluding already-tokenized `*.token.*` files.

    pattern=None matches both the current and legacy annotated namings.
    """
    data_dir = ensure_dirpath(resolve_path(data_dir))
    patterns = (pattern,) if pattern else ANNOTATED_GLOBS
    found = set()
    for pat in patterns:
        found.update(f for f in _glob.glob(os.path.join(data_dir, pat))
                     if '.token.' not in os.path.basename(f)
                     and not f.endswith('.muat.tsv'))
    return sorted(found)


def collect_tokens(files, pos_bin_size=DEFAULT_POS_BIN_SIZE,
                   which=('pos', 'motif', 'ges'), verbose=True, allele_labels=False):
    """Scan the corpus once, returning the observed key sets. Never writes to `files`.

    With allele_labels, also returns seq -> Counter(allele-derived class) so a label can be
    assigned to motifs the baseline dictionary does not contain.
    """
    motif, pos, ges = set(), set(), set()
    allele = {}
    needed = ['chrom', 'pos', 'seq', 'genic', 'exonic', 'strand']
    if allele_labels:
        needed = needed + ['ref', 'alt']

    for i, path in enumerate(files, 1):
        df = pd.read_csv(path, sep='\t', compression='infer', low_memory=False)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(
                f"{path} is missing column(s) {missing}. Expected an ANNOTATED file "
                f"(*.annotate.tsv[.gz] or legacy *.gc.genic.exonic.cs.tsv[.gz]) -- not a raw VCF, not tokenized.")

        if 'motif' in which:
            motif.update(df['seq'].astype(str).unique())
            if allele_labels:
                # A given motif string can arise from variants of different classes, so
                # tally rather than overwrite; the majority wins and disagreements are
                # reported.
                for seq, r, a in zip(df['seq'].astype(str), df['ref'], df['alt']):
                    mt = classify_from_alleles(r, a)
                    if mt is None:
                        continue
                    allele.setdefault(seq, {})
                    allele[seq][mt] = allele[seq].get(mt, 0) + 1
        if 'pos' in which:
            bins = (df['pos'].astype('int64') // int(pos_bin_size)).astype(str)
            pos.update((df['chrom'].astype(str) + '_' + bins).unique())
        if 'ges' in which:
            ges.update((df['genic'].astype(str) + '_' + df['exonic'].astype(str)
                        + '_' + df['strand'].astype(str)).unique())

        if verbose and (i % 100 == 0 or i == len(files)):
            print(f"  scanned {i}/{len(files)} files"
                  f"  motif={len(motif)} pos={len(pos)} ges={len(ges)}", flush=True)

    return motif, pos, ges, allele


def _write_motif(seqs, out_path, labels, mode='inherit', allele=None):
    """dictMutation.tsv -- seq/triplettoken/mut_type, contiguous blocks per mut_type.

    mode:
      inherit  label from `labels` only; motifs absent from it are EXCLUDED.
      hybrid   label from `labels` where known, else from allele lengths.
      refalt   label everything from allele lengths, ignoring `labels`.

    Returns (rows, unlabelled, derived, conflicts).
    """
    allele = allele or {}

    def from_alleles(seq):
        tally = allele.get(seq)
        if not tally:
            return None, False
        best = max(tally.items(), key=lambda kv: (kv[1], kv[0]))[0]
        return best, len(tally) > 1

    by_type, unlabelled, derived, conflicts = {}, [], [], []
    for seq in sorted(seqs):
        mt = None
        if mode in ('inherit', 'hybrid'):
            mt = labels.get(seq)
        if mt is None and mode in ('hybrid', 'refalt'):
            mt, clash = from_alleles(seq)
            if mt is not None:
                derived.append(seq)
                if clash:
                    conflicts.append((seq, dict(allele[seq])))
        if mt is None:
            unlabelled.append(seq)
            continue
        by_type.setdefault(mt, []).append(seq)

    extra = sorted(set(by_type) - set(MUT_TYPE_ORDER))
    rows, token = [], 1
    for mt in MUT_TYPE_ORDER + extra:
        for seq in by_type.get(mt, []):
            rows.append({'seq': seq, 'triplettoken': token, 'mut_type': mt})
            token += 1
    pd.DataFrame(rows, columns=['seq', 'triplettoken', 'mut_type']).to_csv(
        out_path, sep='\t', index=False)
    return rows, unlabelled, derived, conflicts


def _write_motif_provenance(dict_path, mode, labels_path, derived, conflicts):
    """Record how mut_type was assigned.

    The TSV schema is fixed at seq/triplettoken/mut_type -- tokenizing() merges the whole
    dictionary into every output file, so adding a column would change the .muat.tsv schema.
    Hence a sidecar rather than an extra column or a comment line.
    """
    out = dict_path.replace('.tsv', '.provenance.txt')
    with open(out, 'w') as f:
        f.write("mut_type assignment for {}\n".format(os.path.basename(dict_path)))
        f.write("motif_labels mode : {}\n".format(mode))
        f.write("baseline dictionary: {}\n".format(labels_path or '(not used)'))
        f.write("\n")
        f.write("Motifs labelled from ref/alt allele lengths (len(ref)==len(alt)==1 -> SNV,\n"
                "equal length >1 -> MNV, otherwise indel). This answers 'what variant is\n"
                "this', which is NOT the same question the shipped dictionary answers ('what\n"
                "kind of context is this motif'): the shipped labels are corpus-derived and\n"
                "account for neighbouring mutations inside the context window, so the two\n"
                "conventions disagree. SV, MEI and Neg cannot be produced this way.\n"
                "A model trained on this dictionary must be used with THIS dictionary; do not\n"
                "mix it with the pretrained checkpoints.\n\n")
        f.write("{} motif(s) labelled from alleles:\n".format(len(derived)))
        for s in derived:
            f.write("  {}\n".format(s))
        if conflicts:
            f.write("\n{} motif(s) had disagreeing classes across rows "
                    "(majority used):\n".format(len(conflicts)))
            for s, tally in conflicts:
                f.write("  {}\t{}\n".format(s, tally))
    return out


def _write_pos(pos_keys, out_path):
    """dictChpos.tsv -- chrompos/postoken, karyotype order then ascending bin."""
    def key(cp):
        chrom, _, b = str(cp).rpartition('_')
        try:
            b = int(b)
        except ValueError:
            b = -1
        return (_chrom_sort_key(chrom), b)

    rows = [{'chrompos': cp, 'postoken': i}
            for i, cp in enumerate(sorted(pos_keys, key=key), 1)]
    pd.DataFrame(rows, columns=['chrompos', 'postoken']).to_csv(
        out_path, sep='\t', index=False)
    return rows


def _write_ges(ges_keys, out_path):
    """dictGES.tsv -- ges/gestoken."""
    rows = [{'ges': g, 'gestoken': i} for i, g in enumerate(sorted(ges_keys), 1)]
    pd.DataFrame(rows, columns=['ges', 'gestoken']).to_csv(
        out_path, sep='\t', index=False)
    return rows


def _shipped(name):
    return os.path.join(ensure_dirpath(pkg_path('extfile')), name)


def build_dictionaries(data_dir=None, out_dir=None, pos_bin_size=DEFAULT_POS_BIN_SIZE,
                       which=('pos', 'motif', 'ges'), pattern=ANNOTATED_GLOB,
                       suffix='', mut_type_from=None, tokenize_to=None,
                       motif_labels='inherit', files=None, verbose=True):
    """Build the requested dictionaries from the annotated files in data_dir.

    Writes dict{Chpos,Mutation,GES}<suffix>.tsv into out_dir; returns {kind: path}.

    With tokenize_to set, the same corpus is then tokenized into that directory using the
    dictionaries just built (falling back to the shipped ones for any kind not rebuilt).
    Dictionary construction is a corpus-level step -- every sample must be seen before
    token ids can be assigned -- so it cannot be folded into the per-sample `preprocess`
    pass, but it can absorb the tokenizing pass that follows it.
    """
    which = tuple(which)
    unknown = [w for w in which if w not in ('pos', 'motif', 'ges')]
    if unknown:
        raise ValueError(f"unknown dictionary kind(s): {unknown}. Choose from pos, motif, ges.")

    # Either an explicit file list (callers that already resolved their inputs, e.g.
    # `preprocess --preannotated --build-dictionary`, whose inputs may span directories) or a
    # directory to glob.
    if files is not None:
        files = [resolve_path(f) for f in files]
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f"annotated file(s) not found: {missing[:5]}")
        source_desc = f"{len(files)} file(s) given explicitly"
    else:
        if data_dir is None:
            raise ValueError("build_dictionaries needs either data_dir or files")
        files = find_annotated_files(data_dir, pattern)
        source_desc = str(data_dir)
        if not files:
            raise FileNotFoundError(
                f"no annotated files matching '{pattern}' in {data_dir} "
                f"(tokenized *.token.* files are skipped on purpose)")
    if out_dir is None:
        raise ValueError("build_dictionaries needs out_dir")

    if motif_labels not in MOTIF_LABEL_MODES:
        raise ValueError(f"--motif-labels must be one of {MOTIF_LABEL_MODES}, "
                         f"got {motif_labels!r}")

    labels, labels_path = ({}, None)
    if 'motif' in which and motif_labels in ('inherit', 'hybrid'):
        labels, labels_path = load_mut_type_labels(mut_type_from)

    out_dir = ensure_dirpath(resolve_path(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print(f"building {', '.join(which)} from {len(files)} annotated files "
              f"({source_desc})")
        print(f"  pos_bin_size : {pos_bin_size}")
        if 'motif' in which:
            print(f"  motif labels : {motif_labels}")
        if labels_path:
            print(f"  mut_type from: {labels_path} ({len(labels)} labelled motifs)")

    need_alleles = 'motif' in which and motif_labels in ('hybrid', 'refalt')
    motif, pos, ges, allele = collect_tokens(files, pos_bin_size, which, verbose,
                                             allele_labels=need_alleles)

    written = {}
    if 'pos' in which:
        p = os.path.join(out_dir, f'dictChpos{suffix}.tsv')
        rows = _write_pos(pos, p)
        written['pos'] = p
        if verbose:
            print(f"  wrote {p}  ({len(rows)} tokens)")
    if 'motif' in which:
        p = os.path.join(out_dir, f'dictMutation{suffix}.tsv')
        rows, unlabelled, derived, conflicts = _write_motif(
            motif, p, labels, mode=motif_labels, allele=allele)
        written['motif'] = p
        if verbose:
            counts = {mt: sum(1 for r in rows if r['mut_type'] == mt) for mt in MUT_TYPE_ORDER}
            print(f"  wrote {p}  ({len(rows)} tokens)")
            print("    per mut_type: " + ", ".join(f"{k}={v}" for k, v in counts.items() if v))
        if derived:
            print(f"  {len(derived)} motif(s) labelled from ref/alt allele lengths"
                  + (" (baseline dictionary not consulted)" if motif_labels == 'refalt' else
                     " because the baseline dictionary does not contain them")
                  + f", e.g. {derived[:5]}")
            if conflicts:
                print(f"    {len(conflicts)} of those had DISAGREEING classes across rows; "
                      f"the majority was used, e.g. {conflicts[:3]}")
            _write_motif_provenance(p, motif_labels, labels_path, derived, conflicts)
            print("    provenance written to " + p.replace('.tsv', '.provenance.txt'))
        if unlabelled:
            print(f"  WARNING: {len(unlabelled)} observed motif(s) could not be labelled and "
                  f"were EXCLUDED, e.g. {unlabelled[:5]}.")
            if motif_labels == 'inherit':
                print("           mut_type cannot be inferred from a motif string, so under "
                      "--motif-labels inherit")
                print("           anything absent from the baseline is dropped. Use "
                      "--motif-labels hybrid to")
                print("           label these from ref/alt allele lengths instead.")
            else:
                print("           These carry no usable ref/alt (multi-allelic ALT, or absent "
                      "from the corpus).")
        absent = sorted(REQUIRED_MUT_TYPES - {r['mut_type'] for r in rows})
        if absent:
            print(f"  NOTE: this motif dictionary has no {absent} tokens, because the corpus")
            print("        contains no such mutations (SV/MEI come from the separate")
            print("        consensus_sv callset, Neg from negative sampling).")
            print("        This is FINE as long as --mutation-type only names classes that are")
            print("        present: the motif vocabulary is sized from the per-class counts the")
            print("        chosen ratio selects, so e.g. snv+mnv needs only SNV and MNV. Asking")
            print("        for a class with no tokens would size that block to zero.")
            print("        Motifs do not depend on the genome build, so for a new reference")
            print("        --which pos alone is usually enough.")
    if 'ges' in which:
        p = os.path.join(out_dir, f'dictGES{suffix}.tsv')
        rows = _write_ges(ges, p)
        written['ges'] = p
        if verbose:
            print(f"  wrote {p}  ({len(rows)} tokens)")

    # Resolve the full trio: rebuilt where asked for, shipped (or --mut-type-from) otherwise.
    # Tokenizing must use a consistent set, so all three are always named explicitly.
    trio = {
        'pos':   written.get('pos')   or _shipped('dictChpos.tsv'),
        'motif': written.get('motif') or (resolve_path(mut_type_from) if mut_type_from
                                          else _shipped('dictMutation.tsv')),
        'ges':   written.get('ges')   or _shipped('dictGES.tsv'),
    }

    if tokenize_to:
        # Imported here rather than at module scope: preprocessing pulls in the reader and
        # the whole annotation stack, which build-dictionary alone does not need.
        from .preprocessing import preprocessing_annotated_tokenizing
        tokenize_to = ensure_dirpath(resolve_path(tokenize_to))
        os.makedirs(tokenize_to, exist_ok=True)
        if verbose:
            print(f"tokenizing {len(files)} file(s) into {tokenize_to} using:")
            for k in ('motif', 'pos', 'ges'):
                print(f"  {k:5s} {trio[k]}")
        preprocessing_annotated_tokenizing(
            input_files=files,
            tmp_dir=tokenize_to,
            dict_motif=pd.read_csv(trio['motif'], sep='\t'),
            dict_pos=pd.read_csv(trio['pos'], sep='\t'),
            dict_ges=pd.read_csv(trio['ges'], sep='\t'),
        )
        produced = sorted(_glob.glob(os.path.join(tokenize_to, '*.muat.tsv')))
        if not produced:
            raise RuntimeError(f"tokenizing produced no *.muat.tsv in {tokenize_to}")
        if verbose:
            print(f"  wrote {len(produced)} tokenized file(s)")
        written['tokenized'] = tokenize_to
    elif verbose:
        print("Next, tokenize and train with these exact three paths (or re-run with "
              "--tokenize --tmp-dir <dir> to do the tokenizing here):")
        for k in ('motif', 'pos', 'ges'):
            print(f"  --{k if k != 'motif' else 'motif'}-dictionary-filepath {trio[k]}"
                  .replace('--pos-dictionary', '--position-dictionary'))
    return written
