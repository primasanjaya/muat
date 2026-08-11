"""`muat reproduce` / `muat fetch` — run the pinned experiments from
experiments.md offline-safely.

Design (see experiments.md "Reproducing these experiments"):

* Recipes + asset manifest live in the shipped ``pkg_reproduce/experiments.json``.
* Asset download (``fetch_tag``) is separated from execution (``reproduce``) so a
  run can happen on an offline compute node after staging assets on a login node.
* A run resolves every input from the cache directory and never touches the
  network; if something is missing it tells the user the exact ``muat fetch`` line.

This module never imports ``muat.core`` (core imports it); core orchestrates the
actual predict/train dispatch and reuses its existing ``_run_predict``.
"""

import argparse
import hashlib
import json
import os
import tarfile
import urllib.request

from muat._resources import pkg_path
from muat.util import ensure_dirpath, resolve_path


# --------------------------------------------------------------------------- #
# Recipe / config loading
# --------------------------------------------------------------------------- #

def load_experiments():
    """Load the shipped recipe + manifest file."""
    cfg_path = pkg_path('pkg_reproduce', 'experiments.json')
    with open(cfg_path) as f:
        return json.load(f)


def get_recipe(tag, experiments=None):
    """Return the recipe for ``tag`` with defaults merged in, or error."""
    if experiments is None:
        experiments = load_experiments()
    tags = experiments.get('tags', {})
    if tag not in tags:
        raise ValueError(
            "unknown reproduce tag {!r}. Available: {}. "
            "Run `muat reproduce --list` for details.".format(
                tag, ', '.join(sorted(tags))))
    recipe = dict(experiments.get('defaults', {}))
    recipe.update(tags[tag])
    recipe['tag'] = tag
    return recipe


def list_tags(experiments=None):
    """Print a one-line summary per available tag."""
    if experiments is None:
        experiments = load_experiments()
    tags = experiments.get('tags', {})
    print('Available reproduce tags:')
    for tag in sorted(tags):
        r = tags[tag]
        print('  {:<4} [{}/{:<10}] {:<9} {}'.format(
            tag,
            r.get('group', '?'),
            r.get('access', '?'),
            r.get('mode', '?'),
            r.get('purpose', '')))


# --------------------------------------------------------------------------- #
# Cache directory
# --------------------------------------------------------------------------- #

def resolve_cache_dir(cache_dir_arg=None):
    """Resolve the asset cache: --cache-dir > $MUAT_CACHE > ~/.cache/muat.

    On HPC this must point at a filesystem shared between the (online) login
    node used for ``muat fetch`` and the (possibly offline) compute node used
    for ``muat reproduce``.
    """
    cache_dir = (cache_dir_arg or os.environ.get('MUAT_CACHE')
                 or os.path.join(os.path.expanduser('~'), '.cache', 'muat'))
    return ensure_dirpath(resolve_path(cache_dir))


def _assets_dir(cache_dir):
    return ensure_dirpath(os.path.join(cache_dir, 'assets'))


def _manifest_path(cache_dir):
    return os.path.join(cache_dir, 'manifest.json')


def _read_manifest(cache_dir):
    path = _manifest_path(cache_dir)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _write_manifest(cache_dir, manifest):
    with open(_manifest_path(cache_dir), 'w') as f:
        json.dump(manifest, f, indent=2)


# --------------------------------------------------------------------------- #
# Asset acquisition + integrity
# --------------------------------------------------------------------------- #

def _sha256(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _verify_sha256(path, expected):
    """Verify a file's checksum. ``TODO``/None means 'not pinned yet' -> skip."""
    if not expected or expected == 'TODO':
        print('  WARNING: no sha256 pinned for {} — skipping integrity check'
              .format(os.path.basename(path)))
        return
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            'checksum mismatch for {}\n  expected {}\n  got      {}'
            .format(path, expected, actual))
    print('  sha256 OK: {}'.format(os.path.basename(path)))


def _download_url(url, dest, asset=None):
    # The message is written for someone who installed muat from bioconda/PyPI, not
    # for a maintainer: telling them to edit experiments.json inside site-packages is
    # not an action they should take. An unset url means this release was tagged
    # before the data archive was published, which is a property of the release.
    if not url or url == 'TODO':
        raise ValueError(
            'no published download URL for asset {!r} in this muat release, so it '
            'cannot be staged automatically.\n'
            'This release was tagged before the corresponding data/checkpoint archive '
            'was deposited. Either point --cache-dir at a copy you already have, or see '
            'https://github.com/primasanjaya/muat for the archive DOI and the muat '
            'version whose manifest includes it.'
            .format(asset or os.path.basename(dest)))
    print('  downloading {} -> {}'.format(url, dest))
    urllib.request.urlretrieve(url, dest)


def _needed_asset_names(recipe, from_raw):
    """Which manifest assets a tag needs, given the run mode."""
    names = []
    if recipe.get('checkpoint'):              # predict tags reference a checkpoint
        names.append(recipe['checkpoint'])
    if from_raw:
        if recipe.get('raw_data'):
            names.append(recipe['raw_data'])
        if recipe.get('reference'):           # raw inputs need the reference genome
            names.append(recipe['reference'])
    elif recipe.get('data_bundle'):
        names.append(recipe['data_bundle'])
    return list(dict.fromkeys(names))         # de-dup, preserve order


def fetch_tag(tag, cache_dir_arg=None, from_raw=False):
    """Download + verify every asset tag needs into the cache, update manifest.

    Intended to run on a node WITH internet access.
    """
    experiments = load_experiments()
    recipe = get_recipe(tag, experiments)
    assets = experiments.get('assets', {})
    cache_dir = resolve_cache_dir(cache_dir_arg)
    adir = _assets_dir(cache_dir)
    manifest = _read_manifest(cache_dir)

    needed = _needed_asset_names(recipe, from_raw)
    if not needed:
        print('tag {} needs no downloadable assets.'.format(tag))
        return cache_dir

    print('fetching assets for {} into {} (from_raw={})'.format(tag, cache_dir, from_raw))
    for name in needed:
        if name not in assets:
            raise ValueError('recipe references unknown asset {!r}'.format(name))
        spec = assets[name]
        kind = spec.get('kind')

        if kind == 'raw_pcawg':
            _fetch_raw_pcawg(spec, cache_dir, manifest, name)
            continue

        # single-file assets (checkpoint, reference, data_bundle)
        dest = os.path.join(adir, spec['filename'])
        if os.path.exists(dest):
            print('  already present: {}'.format(spec['filename']))
        else:
            _download_url(spec.get('url'), dest, asset=name)
        _verify_sha256(dest, spec.get('sha256'))

        record = {'filename': spec['filename'], 'kind': kind,
                  'sha256': spec.get('sha256'), 'path': dest}

        if spec.get('extract'):
            sub = spec.get('extract_subdir', name)
            target = os.path.join(cache_dir, sub)
            if not os.path.isdir(target):
                print('  extracting {} -> {}'.format(spec['filename'], target))
                with tarfile.open(dest) as tf:
                    tf.extractall(path=cache_dir)
            record['extracted_to'] = target

        manifest[name] = record

    _write_manifest(cache_dir, manifest)
    print('fetch complete. manifest: {}'.format(_manifest_path(cache_dir)))
    return cache_dir


def _fetch_raw_pcawg(spec, cache_dir, manifest, name):
    """Fetch raw open-access PCAWG files via the existing ICGC downloader."""
    from muat.download import download_icgc_object_storage  # lazy: avoids heavy import on every CLI call
    subdir = ensure_dirpath(os.path.join(cache_dir, spec.get('subdir', name)))
    print('  downloading raw PCAWG ({} files) -> {}'.format(len(spec.get('files', [])), subdir))
    download_icgc_object_storage(
        data_path=subdir,
        bucket_name=spec.get('bucket', 'icgc25k-open'),
        endpoint_url=spec.get('endpoint_url', 'https://object.genomeinformatics.org'),
        files_to_download=spec.get('files'),
    )
    manifest[name] = {'kind': 'raw_pcawg', 'path': subdir}


# --------------------------------------------------------------------------- #
# Asset resolution at run time (offline)
# --------------------------------------------------------------------------- #

def _asset_file(cache_dir, asset_name, experiments):
    """Absolute path to a single-file asset in the cache, or None if absent."""
    spec = experiments.get('assets', {}).get(asset_name, {})
    fn = spec.get('filename')
    if not fn:
        return None
    p = os.path.join(_assets_dir(cache_dir), fn)
    return p if os.path.exists(p) else None


def ensure_assets_present(recipe, cache_dir, experiments, from_raw):
    """Raise an actionable error if any needed asset is missing from the cache."""
    missing = []
    for name in _needed_asset_names(recipe, from_raw):
        spec = experiments.get('assets', {}).get(name, {})
        if spec.get('kind') == 'raw_pcawg':
            subdir = os.path.join(cache_dir, spec.get('subdir', name))
            if not os.path.isdir(subdir):
                missing.append(name)
        elif spec.get('extract'):
            target = os.path.join(cache_dir, spec.get('extract_subdir', name))
            if not os.path.isdir(target):
                missing.append(name)
        else:
            if _asset_file(cache_dir, name, experiments) is None:
                missing.append(name)
    if missing:
        # Split the two cases apart. Suggesting `muat fetch` for an asset that has no
        # published url just sends the user in a circle: fetch would fail too. Only
        # assets that ARE downloadable get the fetch hint.
        def _no_published_url(spec):
            # raw_pcawg assets are staged through the ICGC object store rather than a
            # plain url, so a missing 'url' key is normal for them and does NOT mean
            # the archive is unpublished. Only url-based assets can be "not deposited yet".
            if spec.get('kind') == 'raw_pcawg':
                return False
            return spec.get('url') in (None, '', 'TODO')

        unpublished = [n for n in missing
                       if _no_published_url(experiments.get('assets', {}).get(n, {}))]
        fetchable = [n for n in missing if n not in unpublished]

        lines = ['missing cached asset(s) for {}: {}'.format(recipe['tag'], ', '.join(missing))]
        if fetchable:
            fetch_cmd = 'muat fetch {} --cache-dir {}'.format(recipe['tag'], cache_dir.rstrip('/'))
            if from_raw:
                fetch_cmd += ' --from-raw'
            lines.append('Stage the downloadable one(s) ({}) on a node with internet:'
                         .format(', '.join(fetchable)))
            lines.append('    ' + fetch_cmd)
        if unpublished:
            lines.append(
                'No published download URL in this muat release for: {}. This release was '
                'tagged before the corresponding archive was deposited, so these cannot be '
                'fetched automatically — point --cache-dir at a copy you already have, or see '
                'https://github.com/primasanjaya/muat for the archive DOI and the muat version '
                'whose manifest includes it.'.format(', '.join(unpublished)))
        raise FileNotFoundError('\n'.join(lines))


def _split_path(recipe, which):
    """Path to a shipped split file (e.g. the test sample list)."""
    rel = recipe.get('splits', {}).get(which)
    if rel is None:
        raise ValueError('recipe {} has no {!r} split'.format(recipe['tag'], which))
    return pkg_path('pkg_reproduce', *rel.split('/'))


def _bundle_dir(recipe, cache_dir, experiments):
    """Cache directory holding the per-sample preprocessed files for a tag."""
    bundle = experiments['assets'][recipe['data_bundle']]
    return os.path.join(cache_dir, bundle.get('extract_subdir', recipe['data_bundle']))


def _resolve_split_rows(recipe, which, cache_dir, experiments):
    """Read a shipped split and remap each prep_path to its full cached path.

    The split files use MuAt's native schema (prep_path, class_name, class_index),
    but prep_path is stored as a *basename only* (e.g. ``<sample>.muat.tsv``) so
    they stay portable. Returns the rows as dicts with prep_path rewritten to the
    absolute path in the data bundle; raises FileNotFoundError if any is missing."""
    import csv
    split_file = _split_path(recipe, which)
    with open(split_file) as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        if 'prep_path' not in (reader.fieldnames or []):
            raise ValueError(
                'split file {} must have a "prep_path" column (MuAt native '
                'format: prep_path<TAB>class_name<TAB>class_index).'.format(split_file))
        rows = [dict(r) for r in reader if r.get('prep_path', '').strip()]
    if not rows:
        raise ValueError('split file {} has no samples.'.format(split_file))

    bundle_dir = _bundle_dir(recipe, cache_dir, experiments)
    missing = []
    for r in rows:
        full = os.path.join(bundle_dir, os.path.basename(r['prep_path'].strip()))
        r['prep_path'] = full
        if not os.path.exists(full):
            missing.append(full)
    if missing:
        raise FileNotFoundError(
            '{} of {} preprocessed sample file(s) missing from bundle {}, e.g. {}'
            .format(len(missing), len(rows), bundle_dir, missing[0]))
    return rows


def _build_predict_input_list(recipe, cache_dir, experiments, result_dir):
    """Resolve the test split against the cached bundle; write an --input-list
    of absolute preprocessed-file paths for core._run_predict."""
    rows = _resolve_split_rows(recipe, 'test', cache_dir, experiments)
    os.makedirs(result_dir, exist_ok=True)
    list_path = os.path.join(result_dir, '_input_list.txt')
    with open(list_path, 'w') as f:
        f.write('\n'.join(r['prep_path'] for r in rows) + '\n')
    return list_path


def _materialize_split(recipe, which, cache_dir, experiments, out_dir):
    """Write a full split TSV (absolute prep_path + class_name + class_index),
    resolved against the cached bundle, that MuAt's training dataloader consumes.
    Returns the written path."""
    import csv
    rows = _resolve_split_rows(recipe, which, cache_dir, experiments)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, '_{}_split.tsv'.format(which))
    with open(out_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['prep_path', 'class_name', 'class_index'],
                           delimiter='\t', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    return out_path


def build_train_namespace(recipe, cache_dir, save_dir, experiments, seed=None):
    """Construct an argparse.Namespace compatible with
    core._run_train_from_scratch, materializing the train/val splits against the
    cached bundle and pulling hyperparameters from the recipe.

    ``seed`` is the seed already resolved by the caller (``--unseeded`` -> None,
    else ``--seed`` or the recipe's pinned value). It is passed through to
    TrainerConfig so the DataLoader generator and worker RNGs are seeded too, not
    just the process-global RNGs.
    """
    hp = recipe.get('hyperparams', {})
    return argparse.Namespace(
        seed=seed,
        command='train',
        subcommand='from-scratch',
        mutation_type=recipe['mutation_type'],
        use_motif=recipe.get('use_motif', True),
        use_position=recipe.get('use_position', False),
        use_ges=recipe.get('use_ges', False),
        train_split_filepath=_materialize_split(recipe, 'train', cache_dir, experiments, save_dir),
        val_split_filepath=_materialize_split(recipe, 'val', cache_dir, experiments, save_dir),
        save_dir=save_dir,
        epoch=hp.get('epoch', 1),
        learning_rate=hp.get('learning_rate', 6e-4),
        batch_size=hp.get('batch_size', 2),
        n_layer=hp.get('n_layer', 1),
        n_head=hp.get('n_head', 8),
        n_emb=hp.get('n_emb', 128),
        mutation_sampling_size=hp.get('mutation_sampling_size', 5000),
        sampling_replacement=hp.get('sampling_replacement', False),
        patience=hp.get('patience', 0),
        lr_patience=hp.get('lr_patience', None),
        lr_factor=hp.get('lr_factor', 0.5),
        min_lr=hp.get('min_lr', 1e-7),
        motif_dictionary_filepath=None,
        position_dictionary_filepath=None,
        ges_dictionary_filepath=None,
    )


def build_predict_namespace(recipe, cache_dir, result_dir, experiments,
                            from_raw=False, relu=False):
    """Construct an argparse.Namespace compatible with core._run_predict."""
    if from_raw:
        raise NotImplementedError(
            '--from-raw run path is not wired yet for predict tags; the '
            'preprocessed bundle is the supported path. `muat fetch --from-raw` '
            'will still stage the raw data.')

    ckpt = _asset_file(cache_dir, recipe['checkpoint'], experiments)
    input_list = _build_predict_input_list(recipe, cache_dir, experiments, result_dir)

    return argparse.Namespace(
        command='predict',
        source='from-checkpoint',
        ckpt_filepath=ckpt,
        input_filepath=None,
        input_list=input_list,
        result_dir=result_dir,
        hg19=None,
        hg38=None,
        tmp_dir=None,
        relu=relu,
        needs_preprocessing=False,
    )
