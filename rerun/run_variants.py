"""Submit (or run) the manuscript variants listed in rerun/variants.yml.

    python -m bio_analysis.rerun.run_variants --group fast --submit [--partition p.large] [--only NAME ...]
    python -m bio_analysis.rerun.run_variants --group fast --local --only synmulti_MSN   # run in this process

Placeholders: {CT} = cell type name from "foreach", {CTID} = its numeric id, {NCT} = number of neuron types.
"""
import argparse
import os
import subprocess
import sys

import yaml

CT_IDS = {'DA': 0, 'LMAN': 1, 'HVC': 2, 'MSN': 3, 'STN': 4, 'TAN': 5, 'GPe': 6, 'GPi': 7, 'LTS': 8, 'INT1': 9, 'INT2': 10,
          'INT3': 11, 'ASTRO': 12, 'OLIGO': 13, 'MICRO': 14, 'MIGR': 15, 'FRAG': 16}  # v6 numbering; v8: INT4 = 12, glia 13-18
REPO = os.environ.get('BIO_ANALYSIS_REPO', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.environ.get('BIO_ANALYSIS_RESULTS')


def expand(v, version='v6'):
    """expand a variants.yml entry with 'foreach' into concrete variants. For v8 the cell-type lists that end with
    INT3 get INT4 appended (index 12), and the numeric ids follow the v8 numbering."""
    if 'foreach' not in v:
        return [v]
    (key, vals), = v['foreach'].items()
    if version == 'v8' and key == 'CT' and 'INT3' in vals and 'INT4' not in vals:
        vals = list(vals) + ['INT4']
    out = []
    for val in vals:
        fmt = {key: val, key + 'ID': CT_IDS.get(val), 'NCT': 13 if version == 'v8' else 12}  # NCT = number of neuron types
        if key == 'TRIPLE':
            parts = val.split('_')
            fmt.update({f'T{i + 1}': CT_IDS[c] for i, c in enumerate(parts)})
            fmt['LEN'] = '200_50' if any(CT_IDS[c] in (0, 1, 2) for c in parts) else '200'
        w = {k: x for k, x in v.items() if k != 'foreach'}
        w['name'] = v['name'].format(**fmt)
        w['published'] = v['published'].format(**fmt)
        w['set'] = {k: str(x).format(**fmt) for k, x in v.get('set', {}).items()}
        w['sub'] = [[a.format(**fmt), b.format(**fmt)] for a, b in v.get('sub', [])]
        out.append(w)
    return out


def variant_args(v):
    args = [v['script']] + [f'{k}={x}' for k, x in v.get('set', {}).items()]
    for a, b in v.get('sub', []):
        args += ['--sub', a, b]
    for a, b in v.get('osub', []):
        args += ['--osub', a, b]
    return args


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', default='fast')
    p.add_argument('--only', nargs='*')
    p.add_argument('--submit', action='store_true')
    p.add_argument('--local', action='store_true')
    p.add_argument('--partition', default='p.large')
    p.add_argument('--cpus', default='8', help='cores per job (overrides the sbatch script; fast variants are single-threaded)')
    p.add_argument('--mem', default='100G')
    p.add_argument('--version', default='v6', help="v6 or v8 cell types")
    p.add_argument('--prep', default=None)
    a = p.parse_args()
    a.prep = a.prep or os.environ['BIO_ANALYSIS_PREP']
    run_root = os.environ['BIO_ANALYSIS_RUN_ROOT']
    export = f'--export=ALL,BIO_ANALYSIS_PREP={a.prep},BIO_ANALYSIS_RESULTS={RESULTS},BIO_ANALYSIS_RUN_ROOT={run_root}'
    if a.version == 'v8':
        export += f',BIO_ANALYSIS_CT_KEY=celltype_pts_{a.version}_analysis'  # the synapse cache must use the v8 partner types
    spec = yaml.safe_load(open(os.path.join(REPO, 'rerun', 'variants.yml')))
    if a.version == 'v8':
        CT_IDS.update({'INT4': 12, 'ASTRO': 13, 'OLIGO': 14, 'MICRO': 15, 'MIGR': 16, 'FRAG': 17, 'OPC': 18})
    variants = [w for v in spec['variants'] if v.get('group', 'fast') == a.group for w in expand(v, a.version)]
    if a.only:
        variants = [v for v in variants if v['name'] in a.only]
    print(f'{len(variants)} variants')
    for v in variants:
        if a.version != 'v6':
            v['set'] = dict({'version': f"'{a.version}'"}, **v.get('set', {}))
            v['published'] = v['published'].replace('j0251v6', f'j0251{a.version}')
            # scripts that read other runs' csvs name the v6 run dirs literally: point them at the v8 runs
            v['osub'] = [['j0251v6', f'j0251{a.version}']]
        args = variant_args(v)
        if a.submit:
            cmd = ['sbatch', '-p', a.partition, '-c', a.cpus, '--mem', a.mem, '-J', v['name'], export,
                   os.path.join(REPO, 'rerun', 'sbatch_run.sh'), '-m', 'bio_analysis.rerun.run_variant'] + args
            r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            print(v['name'], (r.stdout.strip() or r.stderr.strip()).splitlines()[-1])
        elif a.local:
            env = dict(os.environ, BIO_ANALYSIS_PREP=a.prep, BIO_ANALYSIS_RESULTS=RESULTS, BIO_ANALYSIS_RUN_ROOT=run_root)
            r = subprocess.run([os.path.join(REPO, 'rerun', 'run.sh'), '-m', 'bio_analysis.rerun.run_variant'] + args, env=env)
            print(v['name'], 'exit', r.returncode)
        else:
            print(v['name'], ' '.join(args))


if __name__ == '__main__':
    main()
