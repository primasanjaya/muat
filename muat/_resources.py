try:
    from importlib.resources import files as _files
except ImportError:  # Python 3.8
    from importlib_resources import files as _files


def pkg_path(*parts):
    """Filesystem path to a resource inside the installed muat package.
    Replaces pkg_resources.resource_filename('muat', ...)."""
    p = _files('muat')
    for part in parts:
        p = p.joinpath(part)
    return str(p)
