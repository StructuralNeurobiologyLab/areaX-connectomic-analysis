"""Run one analysis script with some of its hard-coded parameters replaced, without editing the file.

    python -m bio_analysis.rerun.run_variant <script.py> [name=python_expr ...] [--sub OLD NEW ...]

Every ``name=expr`` rewrites the first top-level assignment ``    name = ...`` inside the script's
``if __name__ == '__main__':`` block (the RHS is used verbatim, so ``conn_ct=None``, ``cts_for_loading=np.arange(12)``
and ``spiness=2`` all work). ``--sub OLD NEW`` is a plain text substitution applied everywhere (used for the
dated run-directory prefixes in ``f_name``). The modified source is then executed as ``__main__``.
The variant is logged to ``<f_name>/logs/variant.txt`` after the run when the script defined ``f_name``.
"""
import os
import re
import sys


def main():
    argv = sys.argv[1:]
    if not argv:
        sys.exit(__doc__)
    script = argv[0]
    sets, subs, i = [], [], 1
    while i < len(argv):
        if argv[i] == '--sub':
            subs.append((argv[i + 1], argv[i + 2]))
            i += 3
        elif argv[i] == '--osub':  # optional substitution: no error when the substring is absent
            subs.append((argv[i + 1], argv[i + 2], True))
            i += 3
        else:
            k, v = argv[i].split('=', 1)
            sets.append((k, v))
            i += 1
    repo = os.environ.get('BIO_ANALYSIS_REPO', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = script if os.path.isabs(script) else os.path.join(repo, script)
    src = open(path).read()
    for k, v in sets:
        pat = re.compile(r'^(\s+)' + re.escape(k) + r'\s*=\s*.*$', re.M)  # some files have 'name= value'
        m = pat.search(src)
        if m is None:
            sys.exit(f'run_variant: no assignment "{k} = ..." in {script}')
        src = src[:m.start()] + f'{m.group(1)}{k} = {v}  # run_variant' + src[m.end():]
        if k == 'version':  # some scripts pass the literal instead of the variable (connectivity_fraction_per_ct, map_org2axoness_fullcells, ...)
            src = re.sub(r"Analysis_Params\(\s*(version\s*=\s*)?'v6'\s*\)", 'Analysis_Params(version=version)', src)
    for old, new, *optional in subs:
        if old not in src:
            if optional:
                continue
            sys.exit(f'run_variant: substring {old!r} not found in {script}')
        src = src.replace(old, new)
    print(f'run_variant: {script} with {sets} {subs}', flush=True)
    g = {'__name__': '__main__', '__file__': path}
    sys.argv = [path]
    exec(compile(src, path, 'exec'), g)
    f_name = g.get('f_name')
    if isinstance(f_name, str) and os.path.isdir(f_name):
        os.makedirs(os.path.join(f_name, 'logs'), exist_ok=True)
        with open(os.path.join(f_name, 'logs', 'variant.txt'), 'a') as f:
            f.write(f'{script} {sets} {subs}\n')


if __name__ == '__main__':
    main()
