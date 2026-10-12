#!/usr/bin/env python3
"""Small cosmological DM/hydro, build-mode and synchronized-restart checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import bigfile
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
END = .102


def parameters(path, values):
    path.write_text(''.join(f'{key} = {value}\n' for key, value in values.items()))


def execute(command, directory, expected_failure=False):
    env = os.environ | dict(OMP_NUM_THREADS='2', OMP_PROC_BIND='false',
                            OPENBLAS_NUM_THREADS='1', OMP_DYNAMIC='false')
    with (directory / 'run.log').open('w') as log:
        result = subprocess.run(['mpirun', '--oversubscribe', '--bind-to', 'none', '-np', '2',
                                 *map(str, command)], cwd=directory, env=env,
                                stdout=log, stderr=subprocess.STDOUT, timeout=300)
    text = (directory / 'run.log').read_text()
    if expected_failure:
        if result.returncode == 0 or not re.search(r'(requires|needs|without|compiled|build|unknown|undefined).*SIDM|SIDM.*(build|compiled|disabled|unknown|undefined)', text, re.I):
            raise AssertionError(f'Expected a missing-SIDM-build rejection: {directory}')
    elif result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {directory / "run.log"}')
    return text


def read_state(snapshot, hydro, sidm):
    state = {}
    with bigfile.File(str(snapshot)) as data:
        header = data['Header'].attrs
        assert abs(float(np.asarray(header['Time']).flat[0]) - END) < 1e-12
        assert abs(float(np.asarray(header['TimeIC']).flat[0]) - .1) < 1e-12
        expected = np.array([4096 if hydro else 0, 4096, 0, 0, 0, 0])
        np.testing.assert_array_equal(header['TotNumPart'], expected)
        for species in ([0, 1] if hydro else [1]):
            ids = data[f'{species}/ID'][:]
            assert len(np.unique(ids)) == 4096
            order = np.argsort(ids)
            fields = ['ID', 'Mass', 'Position', 'Velocity', 'Potential']
            if species == 0:
                fields += ['Density', 'InternalEnergy']
            elif sidm:
                fields += ['SIDMSupport', 'TimeBinSIDM', 'SIDMLastVisit']
            for field in fields:
                values = data[f'{species}/{field}'][:][order]
                assert np.isfinite(values).all(), (snapshot, species, field)
                state[f'{species}/{field}'] = values
            assert np.all(state[f'{species}/Mass'] > 0)
            if species == 0:
                assert np.all(state['0/Density'] > 0)
                assert np.all(state['0/InternalEnergy'] > 0)
            elif sidm:
                assert np.all(state['1/SIDMSupport'] > 0)
                assert not np.any(state['1/TimeBinSIDM'])
                assert np.all(state['1/SIDMLastVisit'] == 3 * (1 << 46))
    return state


def compare(left, right, label):
    measured = {}
    # Native threaded SPH reductions and float thermodynamic storage vary even
    # between repeated executions of the same binary. Keep DM/clock checks
    # strict and bound only the observed gas numerical reproducibility floor.
    gas_tolerances = {'0/Position': (2e-10, 2e-7),
                      '0/Velocity': (1e-5, 1e-6),
                      '0/InternalEnergy': (1e-4, 1e-7),
                      '0/Density': (2e-7, 0)}
    for field in left.keys() & right.keys():
        a, b = left[field], right[field]
        if a.dtype.kind in 'iu' or field.endswith('/Mass'):
            np.testing.assert_array_equal(a, b, err_msg=f'{label}: {field}')
        else:
            rtol, atol = gas_tolerances.get(field, (2e-12, 2e-12))
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol,
                                       err_msg=f'{label}: {field}')
            measured[field] = dict(max_absolute_difference=float(np.max(np.abs(a - b))),
                                   rtol=rtol, atol=atol)
    return measured


def gadget_parameters(ic, output, hydro, sidm):
    return dict(InitCondFile=ic, OutputDir=output,
                OutputList='0.1005, 0.101, 0.102', TimeMax=END,
                TimeLimitCPU=240, MaxMemSizePerNode=1000,
                Omega0=.288, OmegaLambda=.712, OmegaBaryon=.0472,
                HubbleParam=.7, MassiveNuLinRespOn=0, SplitGravityTimestepsOn=1, TreeGravOn=1,
                Nmesh=32, MaxSizeTimestep=.01, RandomParticleOffset=0,
                HydroOn=int(hydro), DensityOn=int(hydro),
                DensityIndependentSphOn=0, DensityKernelType="cubic", InitGasTemp=100,
                CoolingOn=0, StarformationOn=0, BlackHoleOn=0,
                MetalReturnOn=0, WindOn=0, SnapshotWithFOF=0,
                OutputTimebins=1, SIDMOn=int(sidm), SIDMNeighbors=32,
                SIDMDirectedProbability=.5, SIDMTransportFraction=.2,
                sigma0=10000000, vdSIDMOn=0)


def endpoint(output):
    index, _ = (output / 'Snapshots.txt').read_text().splitlines()[-1].split()
    return output / f'PART_{int(index):03d}'


def run_case(base, ic, binary, hydro, sidm, debug=False, restart_from=None,
             compiled_sidm=True, statistics=None, sigma0=10000000):
    base.mkdir()
    output = base / 'output'
    output.mkdir()
    mode = [2, -1]
    if restart_from:
        # Resume from the middle output with the original cosmological time
        # origin and output schedule. The source checkpoint remains read-only.
        (output / 'PART_001').symlink_to(restart_from / 'PART_001', target_is_directory=True)
        (output / 'Snapshots.txt').write_text('001 0.101\n')
        mode = [2, 1]
    values = gadget_parameters(os.path.relpath(ic, base), "output", hydro, sidm)
    values['sigma0'] = sigma0
    if not compiled_sidm:
        values = {key: value for key, value in values.items()
                  if not key.startswith('SIDM') and key not in ('sigma0', 'vdSIDMOn')}
    parameters(base / 'gadget.param', values)
    log = execute([binary, base / 'gadget.param', *mode], base)
    if debug and sidm:
        rows = []
        source_only = 0
        epoch = 0
        force_count = 0
        for line in log.splitlines():
            if 'Begin Step' in line:
                epoch = int(re.search(r'\(([0-9a-f]+)\)', line)[1], 16)
                force_count = int(re.search(r'Nf = (\d+)', line)[1])
            if 'SIDMDirectIntervals:' in line:
                row = dict(re.findall(r'(\w+)=([^ ]+)', line))
                row['epoch'] = epoch
                rows.append(row)
                source_only += force_count == 0 and int(row['sources']) > 0
        assert rows, base
        if sigma0:
            assert sum(int(row['events']) for row in rows) > 0, base
            assert source_only > 0, f'No source-only events: {base}'
        else:
            assert all(int(row['events']) == 0 and float(row['sumP']) == 0 for row in rows)
            # This isolates native phase sequencing from physical collisions
            # and transport-induced timestep changes.
            limiters = [dict(re.findall(r'(\w+)=([^ ]+)', line))
                        for line in log.splitlines() if 'SIDM local limiters:' in line]
            assert limiters, f'Missing transport diagnostics: {base}'
            counters = {key: sum(int(row[key]) for row in limiters)
                        for key in ('source_transport', 'gravity_transport', 'deadline_tightened')}
            assert not any(counters.values()), (base, counters)
            if statistics is not None:
                statistics['transport_counters'] = counters
        assert all(0 <= float(row['maxTrialP']) <= .5 * (1 + 1e-12) for row in rows)
        assert all(np.isfinite(float(row[key])) for row in rows for key in ('deltaP', 'deltaK', 'sumP'))
        assert int(rows[0]['events']) == 0 and float(rows[0]['sumP']) == 0
        if statistics is not None:
            statistics.update(source_only_epochs=source_only,
                              events=sum(int(row['events']) for row in rows),
                              events_after_checkpoint=sum(int(row['events']) for row in rows
                                                          if row['epoch'] > 2 * (1 << 46)))
    state = read_state(endpoint(output), hydro, sidm)
    with bigfile.File(str(ic)) as initial:
        for species in ([0, 1] if hydro else [1]):
            np.testing.assert_array_equal(state[f'{species}/ID'],
                                          np.sort(initial[f'{species}/ID'][:]))
            mass = np.float32(initial['Header'].attrs['MassTable'][species])
            assert np.all(state[f'{species}/Mass'] == mass)
    return state


def restart_information(continuous, restarted, full_stats, restart_stats):
    def moments(state):
        mass = state['1/Mass'].astype(float)
        velocity = state['1/Velocity'].astype(float)
        return dict(kinetic=float(.5 * np.sum(mass[:, None] * velocity**2)),
                    momentum=np.sum(mass[:, None] * velocity, axis=0).tolist())
    return dict(continuous=full_stats, restarted=restart_stats,
                continuous_moments=moments(continuous),
                restarted_moments=moments(restarted),
                max_support_relative_difference=float(np.max(np.abs(
                    continuous['1/SIDMSupport'] / restarted['1/SIDMSupport'] - 1))))


def zero_cross_section(case, ic, binary, hydro, disabled):
    statistics = {}
    zero = run_case(case / 'zero-cross-section', ic, binary, hydro, True,
                    debug=True, statistics=statistics, sigma0=0)
    return dict(statistics=statistics,
                comparison=compare(disabled, zero, 'zero cross section/native disabled'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory for generated ICs, outputs and logs')
    parser.add_argument('--genic', type=Path, default=ROOT / 'genic/MP-GenIC')
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--debug', type=Path, required=True)
    parser.add_argument('--without-sidm', type=Path, required=True)
    parser.add_argument('--zero-only', type=Path,
                        help='Only check sigma0=0 against disabled outputs in an existing successful suite')
    args = parser.parse_args()
    base = args.output.resolve()
    base.mkdir(parents=True, exist_ok=False)
    binaries = {key: str(getattr(args, key).resolve())
                for key in ('genic', 'release', 'debug', 'without_sidm')}
    report = dict(passed=False, binaries=binaries, cases=[],
                  binary_sha256={key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                 for key, path in binaries.items()})
    if args.zero_only:
        reference = args.zero_only.resolve()
        assert json.loads((reference / 'validation.json').read_text())['passed']
        report['reference_suite'] = str(reference)
    try:
        for hydro in (False, True):
            name = 'hydro' if hydro else 'dm'
            case = base / name
            case.mkdir()
            if args.zero_only:
                previous = reference / name
                (case / 'IC').symlink_to(previous / 'IC', target_is_directory=True)
                disabled = read_state(endpoint(previous / 'disabled/output'), hydro, False)
                report[name + '_zero_cross_section'] = zero_cross_section(
                    case, case / 'IC/IC', binaries['debug'], hydro, disabled)
                report['cases'].append(name)
                continue
            shutil.copy2(ROOT / 'examples/dm-small/class_pk_9.dat', case / 'power.dat')
            parameters(case / 'genic.param', dict(
                OutputDir='IC', FileBase='IC', Ngrid=16, Nmesh=32,
                BoxSize=4000, Redshift=9, Omega0=.288, OmegaLambda=.712,
                OmegaBaryon=.0472, HubbleParam=.7, ProduceGas=int(hydro),
                MakeGlassGas=0, DifferentTransferFunctions=0,
                UsePeculiarVelocity=1, Seed=181170, MaxMemSizePerNode=1000,
                FileWithInputSpectrum='power.dat'))
            execute([binaries['genic'], case / 'genic.param'], case)
            ic = case / 'IC/IC'
            release = run_case(case / 'release', ic, binaries['release'], hydro, True)
            full_stats, restart_stats = {}, {}
            debug = run_case(case / 'debug', ic, binaries['debug'], hydro, True,
                             debug=True, statistics=full_stats)
            report[name + '_release_debug'] = compare(release, debug, f'{name} release/debug')
            restarted = run_case(case / 'restart', ic, binaries['debug'], hydro, True,
                                 debug=True, restart_from=case / 'debug/output',
                                 statistics=restart_stats)
            release_restart = run_case(case / 'restart-release', ic, binaries['release'],
                                       hydro, True, restart_from=case / 'debug/output')
            report[name + '_restart_release_debug'] = compare(
                restarted, release_restart, f'{name} restart release/debug')
            report[name + '_restart_information'] = restart_information(
                debug, restarted, full_stats, restart_stats)
            disabled = run_case(case / 'disabled', ic, binaries['release'], hydro, False)
            absent = run_case(case / 'without-sidm', ic, binaries['without_sidm'], hydro, False, compiled_sidm=False)
            report[name + '_disabled_comparison'] = compare(disabled, absent, f'{name} SIDM disabled')
            report[name + '_zero_cross_section'] = zero_cross_section(
                case, ic, binaries['debug'], hydro, disabled)
            rejection = case / 'reject-missing-sidm'
            rejection.mkdir()
            parameters(rejection / 'gadget.param', gadget_parameters(os.path.relpath(ic, rejection), 'output', hydro, True))
            execute([binaries['without_sidm'], rejection / 'gadget.param', 2, -1],
                    rejection, expected_failure=True)
            report['cases'].append(name)
        report['passed'] = True
    finally:
        (base / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {base / "validation.json"}')


if __name__ == '__main__':
    main()
