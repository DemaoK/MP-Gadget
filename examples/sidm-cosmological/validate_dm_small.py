#!/usr/bin/env python3
"""Run one native dm-small mode at a time, then compare matching outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / 'examples/dm-small'
MODES = ('baseline', 'release', 'debug', 'cdm')


def digest(path):
    with path.open('rb') as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
        return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def execute(command, directory, args, logname='run.log'):
    env = os.environ | dict(OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC='false',
                            OMP_PROC_BIND='false', OPENBLAS_NUM_THREADS='1')
    command = [args.mpirun, '--oversubscribe', '--bind-to', 'none', '-np', str(args.mpi),
               *map(str, command)]
    started = time.monotonic()
    with (directory / logname).open('w') as log:
        process = subprocess.Popen(command, cwd=directory, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            status = 124
    return dict(command=command, returncode=status, wall_seconds=time.monotonic()-started,
                mpi=args.mpi, threads=args.threads, timeout=args.timeout)


def prepare(base, args):
    base.mkdir(parents=True, exist_ok=True)
    for source in [NATIVE / 'paramfile.genic', *sorted(NATIVE.glob('class_pk*'))]:
        destination = base / source.name
        if destination.exists() and digest(destination) != digest(source):
            raise RuntimeError(f'Native input differs: {destination}')
        if not destination.exists():
            shutil.copy2(source, destination)
    if args.copy_only:
        return
    if (base / 'IC/IC').exists():
        raise FileExistsError(f'Shared IC already exists: {base / "IC/IC"}')
    binary = args.genic.resolve(strict=True)
    report = dict(binary=str(binary), sha256=digest(binary),
                  inputs={p.name: digest(p) for p in base.glob('class_pk*')})
    report['paramfile_sha256'] = digest(base / 'paramfile.genic')
    report.update(execute([binary, 'paramfile.genic'], base, args, 'genic.log'))
    if digest(binary) != report['sha256']:
        raise RuntimeError('The selected GenIC binary changed during preparation')
    write_json(base / 'preparation.json', report)
    if report['returncode']:
        raise RuntimeError(f'GenIC failed: {base / "genic.log"}')


def run(base, args):
    if not (base / 'IC/IC').exists():
        raise FileNotFoundError('Run --mode prepare with the frozen MP-GenIC first')
    case = base / (args.name or args.mode)
    case.mkdir(exist_ok=False)
    output = case / 'output'
    output.mkdir()
    binary = args.binary.resolve(strict=True)
    values = dict(InitCondFile='../IC/IC', OutputDir='output', SIDMOn=int(args.mode != 'cdm'),
                  vdSIDMOn=1, sigma0=147.1, wTurn=120, SIDMNeighbors=38,
                  SIDMDirectedProbability=.5, SIDMTransportFraction=args.transport_fraction)
    if args.without_sidm:
        values = {k: v for k, v in values.items()
                  if not k.startswith("SIDM") and k not in ("vdSIDMOn", "sigma0", "wTurn")}
    for key, value in [('TimeLimitCPU', args.cpu_limit), ('MaxMemSizePerNode', args.memory)]:
        if value is not None:
            values[key] = value
    text = (NATIVE / 'paramfile.gadget').read_text()
    for key, value in values.items():
        pattern = rf'(?m)^\s*{key}\s*=.*$'
        if re.search(pattern, text):
            text = re.sub(pattern, f'{key} = {value}', text)
        else:
            text += f'\n{key} = {value}\n'
    (case / 'paramfile.gadget').write_text(text)
    for source in base.glob('class_pk*'):
        (case / source.name).symlink_to(source)
    restart = -1
    if args.restart_from:
        snapshot = args.restart_from.resolve(strict=True)
        match = re.fullmatch(r'PART_(\d+)', snapshot.name)
        if not match:
            raise ValueError('--restart-from must name a native PART_NNN snapshot')
        restart = int(match[1])
        (output / snapshot.name).symlink_to(snapshot, target_is_directory=True)
        pig = snapshot.with_name(snapshot.name.replace('PART', 'PIG'))
        if pig.exists():
            (output / pig.name).symlink_to(pig, target_is_directory=True)
        import bigfile
        import numpy as np
        with bigfile.File(str(snapshot)) as data:
            atime = float(np.asarray(data['Header'].attrs['Time']).flat[0])
        (output / 'Snapshots.txt').write_text(f'{restart:03d} {atime:.17g}\n')
        power = snapshot.parent / f'powerspectrum-{atime:.4f}.txt'
        if power.exists():
            (output / power.name).symlink_to(power)
    report = dict(mode=args.mode, binary=str(binary), sha256=digest(binary),
                  parameter_overrides=values, compiled_sidm=not args.without_sidm,
                  restart_from=str(args.restart_from) if args.restart_from else None)
    report.update(execute([binary, 'paramfile.gadget', 2, restart], case, args))
    if digest(binary) != report['sha256']:
        raise RuntimeError('The selected binary changed during the run')
    write_json(case / 'run.json', report)
    if report['returncode']:
        raise RuntimeError(f'Run failed or timed out ({report["returncode"]}): {case / "run.log"}')
    if args.mode == 'cdm' and not args.restart_from and .25 in snapshots(case):
        with (case / 'native-check.log').open('w') as log:
            status = subprocess.run([sys.executable, NATIVE / 'check_results.py'], cwd=case,
                                    stdout=log, stderr=subprocess.STDOUT).returncode
        if status:
            raise AssertionError(f'Native CDM checks failed: {case / "native-check.log"}')


def snapshots(case):
    index = case / 'output/Snapshots.txt'
    rows = index.read_text().splitlines() if index.exists() else []
    return {float(a): case / f'output/PART_{int(i):03d}' for i, a in (r.split() for r in rows)}


def read_state(path, sidm, atime=None):
    import bigfile
    import numpy as np
    fields = ['ID', 'Mass', 'Position', 'Velocity']
    if sidm:
        fields += ['SIDMSupport', 'TimeBinSIDM', 'SIDMLastVisit']
    with bigfile.File(str(path)) as data:
        if atime is not None:
            assert abs(float(np.asarray(data['Header'].attrs['Time']).flat[0])-atime) < 1e-12
        assert abs(float(np.asarray(data['Header'].attrs['TimeIC']).flat[0])-.1) < 1e-12
        np.testing.assert_array_equal(data['Header'].attrs['TotNumPart'], [0, 64**3, 0, 0, 0, 0])
        state = {field: data[f'1/{field}'][:] for field in fields}
        order = np.argsort(state['ID'])
        state = {field: value[order] for field, value in state.items()}
        assert len(np.unique(state['ID'])) == 64**3
        assert all(np.isfinite(value).all() for value in state.values())
        assert np.all(state['Mass'] > 0)
        if sidm:
            assert np.all(state['SIDMSupport'] > 0)
            assert np.all(state['SIDMLastVisit'] == state['SIDMLastVisit'][0])
            native_epochs = {.15: 1 << 46, .2: 2 << 46, .25: 3 << 46}
            if atime in native_epochs:
                assert np.all(state['SIDMLastVisit'] == native_epochs[atime])
        return state


def timers(case):
    """Read cumulative root seconds, retaining the original printed a keys."""
    blocks, atime = {}, None
    lines = []
    for path in sorted((case / 'output').glob('cpu.txt*')):
        lines.extend(path.read_text().splitlines())
    for line in lines:
        match = re.search(r'Step\s+\d+.*Time:\s*([\d.eE+-]+)', line)
        if match:
            atime = match[1]
        elif atime:
            fields = line.split()
            # Root has no label: six values, beginning with cumulative mean.
            if len(fields) == 6 and fields[1].endswith('%') and fields[3].endswith('%'):
                blocks[atime] = float(fields[0])
    return blocks


def analyze(base):
    import bigfile
    import numpy as np
    cases = {p.name: p for p in base.iterdir() if (p / 'run.json').exists()}
    report = dict(cases={}, comparisons={},
                  halo_comparison_note='Halo comparisons use ranked masses, not matched individual halos.')
    with bigfile.File(str(base / 'IC/IC')) as initial:
        initial_ids = np.sort(initial['1/ID'][:])
        initial_mass = np.float32(initial['Header'].attrs['MassTable'][1])
        box_mpc_h = float(np.asarray(initial['Header'].attrs['BoxSize']).flat[0])/1000
    particle_nyquist = np.pi*64/box_mpc_h
    report['particle_nyquist_h_Mpc'] = particle_nyquist
    for name, case in cases.items():
        meta = json.loads((case / 'run.json').read_text())
        outputs = snapshots(case)
        for a, path in outputs.items():
            current = read_state(path, meta['mode'] != 'cdm', a)
            np.testing.assert_array_equal(current['ID'], initial_ids)
            assert np.all(current['Mass'] == initial_mass)
        diagnostics = [dict(re.findall(r'(\w+)=([^ ]+)', line))
                       for line in (case / 'run.log').read_text().splitlines()
                       if 'SIDMDirectIntervals:' in line]
        for row in diagnostics:
            assert int(row['N']) == 64**3 and 0 <= int(row['sources']) <= 64**3
            assert 0 <= float(row['maxTrialP']) <= .5*(1+1e-12)
            assert all(np.isfinite(float(row[k])) for k in ('deltaK', 'deltaP', 'sumP'))
        report['cases'][name] = dict(**meta, scale_factors=sorted(outputs), root_seconds=timers(case),
                                   complete=.25 in outputs, diagnostics=len(diagnostics),
                                   events=sum(int(row['events']) for row in diagnostics))
    pairs = [('baseline', 'release'), ('release', 'debug'),
             ('release-restart', 'debug-restart'), ('cdm', 'cdm-runtime')]
    pairs += [('release', name) for name in cases if name != 'release' and
              report['cases'][name]['mode'] == 'release' and not name.endswith('-restart')]
    for left, right in pairs:
        if left not in cases or right not in cases:
            continue
        aout, bout = snapshots(cases[left]), snapshots(cases[right])
        common = sorted(aout.keys() & bout.keys())
        compared = {}
        for a in common:
            sidm = report['cases'][left]['mode'] != 'cdm'
            x, y = read_state(aout[a], sidm, a), read_state(bout[a], sidm, a)
            np.testing.assert_array_equal(x['ID'], y['ID'])
            np.testing.assert_array_equal(x['Mass'], y['Mass'])
            metrics = {}
            if (left.startswith('release') and right.startswith('debug')) or not sidm:
                for field in x:
                    if x[field].dtype.kind in 'iu':
                        np.testing.assert_array_equal(x[field], y[field], err_msg=f'a={a}, {field}')
                    else:
                        np.testing.assert_allclose(x[field], y[field], rtol=2e-12, atol=2e-12,
                                                   err_msg=f'a={a}, {field}')
                    metrics[field] = float(np.max(np.abs(x[field].astype(float)-y[field])))
            power_paths = [cases[n] / f'output/powerspectrum-{a:.4f}.txt' for n in (left, right)]
            if all(p.exists() for p in power_paths):
                powers = [np.loadtxt(p) for p in power_paths]
                assert all(np.isfinite(p).all() for p in powers)
                np.testing.assert_array_equal(powers[0][:, 0], powers[1][:, 0])
                good = (powers[0][:, 0] > 0) & (powers[0][:, 1] > 0)
                metrics['power_max_fractional_difference'] = float(np.max(np.abs(powers[1][good, 1]/powers[0][good, 1]-1)))
                resolved = good & (powers[0][:, 0] <= particle_nyquist)
                metrics['power_max_fractional_difference_below_particle_nyquist'] = float(np.max(np.abs(
                    powers[1][resolved, 1]/powers[0][resolved, 1]-1)))
            elif aout[a].is_symlink() or bout[a].is_symlink():
                metrics['power_comparison'] = 'skipped_missing_restart_seed_spectrum'
            else:
                raise FileNotFoundError(f'Missing output power at a={a}')
            halos = []
            for path in (aout[a], bout[a]):
                pig = path.with_name(path.name.replace('PART', 'PIG'))
                if not pig.exists() and path.is_symlink():
                    metrics['halo_comparison'] = 'skipped_missing_restart_seed_catalogue'
                    break
                with bigfile.File(str(pig)) as data:
                    masses = data['FOFGroups/Mass'][:]
                    assert np.isfinite(masses).all() and np.all(masses > 0)
                    halos.append(np.sort(masses)[::-1])
            if len(halos) == 2:
                count = min(30, len(halos[0]), len(halos[1]))
                metrics['halo_counts'] = [len(m) for m in halos]
                metrics['largest_fof_particle_counts'] = {name: int(round(float(masses[0]/initial_mass)))
                    if len(masses) else 0 for name, masses in zip((left, right), halos)}
                metrics['top_halo_max_fractional_difference'] = float(np.max(np.abs(halos[1][:count]/halos[0][:count]-1))) if count else None
            compared[str(a)] = metrics
        ta, tb = timers(cases[left]), timers(cases[right])
        report['comparisons'][f'{left}/{right}'] = dict(outputs=compared,
            state_comparison='checked' if common else 'pending_no_matching_snapshots',
            matched_root_seconds={a: {left: ta[a], right: tb[a], 'ratio': tb[a]/ta[a]}
                                  for a in ta.keys() & tb.keys() if ta[a] > 0})
    write_json(base / 'comparison.json', report)
    print(base / 'comparison.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--mode', choices=('prepare', *MODES, 'analyze'), required=True)
    parser.add_argument('--genic', type=Path, default=ROOT / 'genic/MP-GenIC')
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--name')
    parser.add_argument('--restart-from', type=Path)
    parser.add_argument('--copy-only', action='store_true')
    parser.add_argument('--mpi', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--mpirun', default='mpirun')
    parser.add_argument('--timeout', type=float, default=7200)
    parser.add_argument('--cpu-limit', type=int)
    parser.add_argument('--memory', type=int)
    parser.add_argument('--transport-fraction', type=float, default=.2)
    parser.add_argument('--without-sidm', action='store_true',
                        help='CDM control binary compiled without SIDM; omit SIDM parameter keys')
    args = parser.parse_args()
    base = args.workdir.resolve()
    if ROOT / 'build' not in base.parents:
        parser.error('--workdir must be an ignored directory beneath this checkout build/')
    if args.mode in MODES and not args.binary:
        parser.error('--binary is required for individual runs')
    if args.without_sidm and args.mode != 'cdm':
        parser.error('--without-sidm is only valid with --mode cdm')
    if not 0 < args.transport_fraction <= .25:
        parser.error('--transport-fraction must be in (0, .25]')
    if min(args.mpi, args.threads, args.timeout) <= 0:
        parser.error('MPI ranks, threads and timeout must be positive')
    if args.mode == 'prepare':
        prepare(base, args)
    elif args.mode == 'analyze':
        analyze(base)
    else:
        run(base, args)


if __name__ == '__main__':
    main()
