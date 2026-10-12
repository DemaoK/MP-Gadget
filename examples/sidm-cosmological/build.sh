#!/bin/bash
# Build isolated dependencies and three reproducible executable variants.
set -euo pipefail
worktree="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source_repo="${1:?usage: build.sh SOURCE_CHECKOUT_WITH_INSTALLED_DEPENDENCIES [JOBS]}"
jobs="${2:-4}"
python - "$worktree" "$source_repo" "$jobs" <<'PY'
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

root, source = (Path(value).resolve() for value in sys.argv[1:3])
jobs = sys.argv[3]
output = root / 'build/sidm-cosmological'
output.mkdir(parents=True, exist_ok=True)
for directory in ('include', 'lib'):
    target = root / 'depends' / directory
    if target.is_symlink():
        raise SystemExit(f'Refusing shared writable dependencies: {target}')
    if not target.exists():
        shutil.copytree(source / 'depends' / directory, target, symlinks=False)
options = root / 'Options.mk'
if options.is_symlink():
    raise SystemExit('Options.mk must be a local file')
manifest = {}
commands = []
for mode, defines in (('without-sidm', ''), ('debug', '-DSIDM -DSIDMDEBUG'),
                      ('release', '-DSIDM')):
    config = f'OPTIMIZE = -fopenmp -O2 -g -Wall\nLOW_PRECISION = double\nOPT = {defines}\n'
    options.write_text(config)
    (output / f'Options-{mode}.mk').write_text(config)
    with (output / f'build-{mode}.log').open('w') as log:
        for directory in ('libgadget', 'gadget', 'libgenic', 'genic'):
            command = ['make', '-C', str(root / directory), f'-j{jobs}']
            commands.append(dict(mode=mode, command=command))
            subprocess.run(command,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    binary = output / f'MP-Gadget-{mode}'
    shutil.copy2(root / 'gadget/MP-Gadget', binary)
    manifest[binary.name] = hashlib.sha256(binary.read_bytes()).hexdigest()
shutil.copy2(root / 'genic/MP-GenIC', output / 'MP-GenIC')
manifest['MP-GenIC'] = hashlib.sha256((output / 'MP-GenIC').read_bytes()).hexdigest()
(output / 'binaries.json').write_text(json.dumps(manifest, indent=2) + '\n')
listed = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z',
                                  '--cached', '--others', '--exclude-standard'])
sources = {}
for value in listed.split(b'\0'):
    if not value:
        continue
    relative = Path(os.fsdecode(value))
    path = root / relative
    if path.is_file() and (relative.suffix in ('.c', '.h', '.mk', '.sh', '.py')
                           or relative.name.startswith('Makefile')):
        sources[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
record = dict(revision=subprocess.check_output(['git', '-C', str(root), 'rev-parse',
                                                'HEAD'], text=True).strip(),
              commands=commands, source_sha256=sources, binaries=manifest,
              allowed_cpus=sorted(os.sched_getaffinity(0)), jobs=int(jobs))
(output / 'build-record.json').write_text(json.dumps(record, indent=2) + '\n')
print(output)
PY
