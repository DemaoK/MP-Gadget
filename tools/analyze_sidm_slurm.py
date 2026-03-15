#!/usr/bin/env python3
"""Analyze SIDM diagnostics from an MP-Gadget SLURM log.

This parser is designed for very large text logs (multi-GB) and streams line by
line. It extracts per-step metrics and writes:
1) SIDM computation cost compared to other runtime components.
2) Collision rate as a function of atime.
3) Suppressed collisions compared to accepted collisions.
4) Momentum-conservation diagnostics from SIDM warning lines.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TIMESTAMP_RE = re.compile(r"\[\s*([0-9]+\.[0-9]+)\s*\]")
STEP_RE = re.compile(
    r"Begin Step\s+(\d+),\s+Time:\s*([0-9eE+\-\.]+)"
)
SCATTER_RE = re.compile(r"SIDM:\s*Particle\s+\d+\s+scatters\s+with\s+\d+")
SUPPRESS_RE = re.compile(
    r"SIDM:\s*Particle\s+\d+\s+(?:suppresses self scatter|self suppresses scatter)\s+with\s+\d+"
)
SIDM_STEP_SUMMARY_RE = re.compile(
    r"SIDM step summary \(a=([0-9eE+\-\.]+)\):\s*"
    r"accepted=(\d+)\s+suppressed=(\d+)\s*"
    r"\(prob-reject=(\d+)\s+claimed-partner=(\d+)\s+threshold=(\d+)\)\s*"
    r"mutual-skip=(\d+)\s+attempted=(\d+)"
)
MOMENTUM_WARN_RE = re.compile(
    r"WARNING:\s*SIDM momentum check residual:\s*"
    r"sum\(m\*SIDMAccel\)=\(([^,]+),([^,]+),([^)]+)\),\s*"
    r"abs-sum=\(([^,]+),([^,]+),([^)]+)\),.*?impacted=(\d+)"
)


@dataclass
class MomentumRecord:
    step: int
    atime: float
    timestamp: float
    sum_x: float
    sum_y: float
    sum_z: float
    abs_x: float
    abs_y: float
    abs_z: float
    impacted: int
    max_component_ratio: float
    l2_ratio: float


@dataclass
class StepRecord:
    step: int
    atime: float
    start_ts: float
    end_ts: float = math.nan
    duration: float = math.nan
    sidm_time: float = 0.0
    gravtree_time: float = 0.0
    domain_exchange_time: float = 0.0
    collisions: int = 0
    suppressed: int = 0
    momentum_warnings: int = 0
    summary_present: int = 0
    step1_prob_reject: int = 0
    suppressed_claimed_partner: int = 0
    suppressed_threshold: int = 0
    mutual_skip: int = 0
    attempted: int = 0


class SidmLogAnalyzer:
    def __init__(self) -> None:
        self.steps: List[StepRecord] = []
        self.momentum: List[MomentumRecord] = []
        self._current_step: Optional[StepRecord] = None
        self._last_ts: Optional[float] = None

        self._sidm_start: Optional[float] = None
        self._grav_start: Optional[float] = None
        self._domain_start: Optional[float] = None

    @staticmethod
    def _parse_float(raw: str) -> Optional[float]:
        try:
            return float(raw.strip())
        except ValueError:
            return None

    @staticmethod
    def _extract_ts(line: str) -> Optional[float]:
        m = TIMESTAMP_RE.search(line)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    @staticmethod
    def _safe_dt(t1: Optional[float], t0: Optional[float]) -> float:
        if t1 is None or t0 is None:
            return 0.0
        return max(0.0, t1 - t0)

    def _close_open_intervals(self, end_ts: Optional[float]) -> None:
        if self._current_step is None:
            self._sidm_start = None
            self._grav_start = None
            self._domain_start = None
            return

        if self._sidm_start is not None:
            self._current_step.sidm_time += self._safe_dt(end_ts, self._sidm_start)
            self._sidm_start = None
        if self._grav_start is not None:
            self._current_step.gravtree_time += self._safe_dt(end_ts, self._grav_start)
            self._grav_start = None
        if self._domain_start is not None:
            self._current_step.domain_exchange_time += self._safe_dt(
                end_ts, self._domain_start
            )
            self._domain_start = None

    def _finalize_current_step(self, end_ts: Optional[float]) -> None:
        if self._current_step is None:
            return
        self._close_open_intervals(end_ts)
        if end_ts is not None:
            self._current_step.end_ts = end_ts
            self._current_step.duration = self._safe_dt(end_ts, self._current_step.start_ts)
        self.steps.append(self._current_step)
        self._current_step = None

    def _handle_new_step(self, step_id: int, atime: float, ts: Optional[float]) -> None:
        if ts is None:
            # Step headers in this log format always include timestamp; if absent,
            # use latest known timestamp to keep continuity.
            ts = self._last_ts
        self._finalize_current_step(ts)
        if ts is None:
            return
        self._current_step = StepRecord(step=step_id, atime=atime, start_ts=ts)

    def _handle_momentum_warning(self, line: str, ts: Optional[float]) -> None:
        if self._current_step is None:
            return
        if ts is None:
            ts = self._last_ts
        if ts is None:
            return

        for m in MOMENTUM_WARN_RE.finditer(line):
            vals = [self._parse_float(m.group(i)) for i in range(1, 7)]
            if any(v is None for v in vals):
                continue
            sum_x, sum_y, sum_z, abs_x, abs_y, abs_z = vals  # type: ignore[assignment]
            try:
                impacted = int(m.group(7))
            except ValueError:
                impacted = -1

            comp_den = max(abs(abs_x), abs(abs_y), abs(abs_z))
            comp_num = max(abs(sum_x), abs(sum_y), abs(sum_z))
            max_component_ratio = comp_num / comp_den if comp_den > 0 else math.nan

            l2_sum = math.sqrt(sum_x * sum_x + sum_y * sum_y + sum_z * sum_z)
            l2_abs = math.sqrt(abs_x * abs_x + abs_y * abs_y + abs_z * abs_z)
            l2_ratio = l2_sum / l2_abs if l2_abs > 0 else math.nan

            self._current_step.momentum_warnings += 1
            self.momentum.append(
                MomentumRecord(
                    step=self._current_step.step,
                    atime=self._current_step.atime,
                    timestamp=ts,
                    sum_x=sum_x,
                    sum_y=sum_y,
                    sum_z=sum_z,
                    abs_x=abs_x,
                    abs_y=abs_y,
                    abs_z=abs_z,
                    impacted=impacted,
                    max_component_ratio=max_component_ratio,
                    l2_ratio=l2_ratio,
                )
            )

    def _handle_step_summary(self, line: str, ts: Optional[float]) -> None:
        if self._current_step is None:
            return
        for m in SIDM_STEP_SUMMARY_RE.finditer(line):
            try:
                accepted = int(m.group(2))
                suppressed = int(m.group(3))
                prob_reject = int(m.group(4))
                claimed_partner = int(m.group(5))
                threshold = int(m.group(6))
                mutual_skip = int(m.group(7))
                attempted = int(m.group(8))
            except ValueError:
                continue

            # Prefer compact per-step debug counters when available.
            self._current_step.summary_present = 1
            self._current_step.collisions = accepted
            self._current_step.suppressed = suppressed
            self._current_step.step1_prob_reject = prob_reject
            self._current_step.suppressed_claimed_partner = claimed_partner
            self._current_step.suppressed_threshold = threshold
            self._current_step.mutual_skip = mutual_skip
            self._current_step.attempted = attempted

            # Close SIDM timing on the explicit per-step summary. Relying on the
            # first non-"SIDM" line is incorrect because treewalk memory-report
            # lines inside SIDM do not contain the SIDM tag.
            if self._sidm_start is not None:
                if ts is None:
                    ts = self._last_ts
                self._current_step.sidm_time += self._safe_dt(ts, self._sidm_start)
                self._sidm_start = None

    def parse(self, log_path: Path) -> None:
        with log_path.open("r", encoding="utf-8", errors="replace") as fobj:
            for line in fobj:
                ts = self._extract_ts(line)
                if ts is not None:
                    self._last_ts = ts

                step_match = STEP_RE.search(line)
                if step_match:
                    try:
                        step_id = int(step_match.group(1))
                        atime = float(step_match.group(2))
                    except ValueError:
                        step_id = -1
                        atime = math.nan
                    self._handle_new_step(step_id, atime, ts)
                    continue

                if self._current_step is None:
                    continue

                self._handle_step_summary(line, ts)
                if self._current_step.summary_present == 0:
                    # Fallback path for older logs without step-summary counters.
                    self._current_step.collisions += len(SCATTER_RE.findall(line))
                    self._current_step.suppressed += len(SUPPRESS_RE.findall(line))
                self._handle_momentum_warning(line, ts)

                if "Treewalk GRAVTREE iter " in line and ts is not None and self._grav_start is None:
                    self._grav_start = ts
                if (
                    self._grav_start is not None
                    and ts is not None
                    and ("GRAVTREE Ngblist:" in line or "Forces computed." in line)
                ):
                    self._current_step.gravtree_time += self._safe_dt(ts, self._grav_start)
                    self._grav_start = None

                if (
                    "Attempting a domain exchange" in line
                    and ts is not None
                    and self._domain_start is None
                ):
                    self._domain_start = ts
                if (
                    self._domain_start is not None
                    and ts is not None
                    and "Done particle data exchange" in line
                ):
                    self._current_step.domain_exchange_time += self._safe_dt(
                        ts, self._domain_start
                    )
                    self._domain_start = None

                if (
                    ts is not None
                    and self._sidm_start is None
                    and (
                        "--------IN SIDM_FORCE --------" in line
                        or "Treewalk SIDM_NGB iter " in line
                        or "Treewalk SIDM_SCATTER iter " in line
                    )
                ):
                    self._sidm_start = ts

        self._finalize_current_step(self._last_ts)


def _as_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _valid_steps(steps: Sequence[StepRecord]) -> List[StepRecord]:
    ret = []
    for st in steps:
        if math.isnan(st.duration) or st.duration <= 0:
            continue
        if math.isnan(st.atime):
            continue
        ret.append(st)
    return ret


def write_step_csv(path: Path, steps: Sequence[StepRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.writer(fobj)
        writer.writerow(
            [
                "step",
                "atime",
                "start_ts",
                "end_ts",
                "step_walltime_s",
                "sidm_walltime_s",
                "gravtree_walltime_s",
                "domain_exchange_walltime_s",
                "collisions",
                "suppressed",
                "momentum_warnings",
                "summary_present",
                "step1_prob_reject",
                "suppressed_claimed_partner",
                "suppressed_threshold",
                "mutual_skip",
                "attempted",
            ]
        )
        for st in steps:
            writer.writerow(
                [
                    st.step,
                    st.atime,
                    st.start_ts,
                    st.end_ts,
                    st.duration,
                    st.sidm_time,
                    st.gravtree_time,
                    st.domain_exchange_time,
                    st.collisions,
                    st.suppressed,
                    st.momentum_warnings,
                    st.summary_present,
                    st.step1_prob_reject,
                    st.suppressed_claimed_partner,
                    st.suppressed_threshold,
                    st.mutual_skip,
                    st.attempted,
                ]
            )


def write_momentum_csv(path: Path, records: Sequence[MomentumRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.writer(fobj)
        writer.writerow(
            [
                "step",
                "atime",
                "timestamp",
                "sum_x",
                "sum_y",
                "sum_z",
                "abs_x",
                "abs_y",
                "abs_z",
                "impacted",
                "max_component_ratio",
                "l2_ratio",
            ]
        )
        for rec in records:
            writer.writerow(
                [
                    rec.step,
                    rec.atime,
                    rec.timestamp,
                    rec.sum_x,
                    rec.sum_y,
                    rec.sum_z,
                    rec.abs_x,
                    rec.abs_y,
                    rec.abs_z,
                    rec.impacted,
                    rec.max_component_ratio,
                    rec.l2_ratio,
                ]
            )


def plot_cost_comparison(steps: Sequence[StepRecord], out: Path) -> None:
    x = _as_array([st.atime for st in steps])
    total = _as_array([st.duration for st in steps])
    sidm = np.clip(_as_array([st.sidm_time for st in steps]), 0.0, None)
    grav = np.clip(_as_array([st.gravtree_time for st in steps]), 0.0, None)
    dom = np.clip(_as_array([st.domain_exchange_time for st in steps]), 0.0, None)
    non_sidm = np.clip(total - sidm, 0.0, None)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(x, total, color="black", lw=1.0, label="Total step walltime")
    axes[0].plot(x, sidm, color="tab:red", lw=1.0, label="SIDM walltime")
    axes[0].plot(x, non_sidm, color="tab:blue", lw=1.0, label="Non-SIDM walltime")
    axes[0].set_ylabel("Seconds per step")
    axes[0].set_title("Per-step cost: SIDM versus the rest of code")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    sidm_tot = float(np.nansum(sidm))
    grav_tot = float(np.nansum(grav))
    dom_tot = float(np.nansum(dom))
    total_tot = float(np.nansum(total))
    other_tot = max(0.0, total_tot - sidm_tot - grav_tot - dom_tot)

    labels = ["SIDM", "GRAVTREE", "DomainExchange", "Other"]
    vals = [sidm_tot, grav_tot, dom_tot, other_tot]
    colors = ["tab:red", "tab:green", "tab:orange", "tab:gray"]
    axes[1].bar(labels, vals, color=colors)
    axes[1].set_ylabel("Integrated seconds")
    axes[1].set_title("Integrated runtime composition (approximate)")
    axes[1].grid(axis="y", alpha=0.25)

    for idx, val in enumerate(vals):
        axes[1].text(idx, val, f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    axes[1].set_xlabel("atime")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_collision_rate(steps: Sequence[StepRecord], out: Path) -> None:
    x = _as_array([st.atime for st in steps])
    collisions = _as_array([st.collisions for st in steps])
    delta_a = np.diff(x, append=np.nan)
    rate_da = np.divide(
        collisions,
        delta_a,
        out=np.full_like(collisions, np.nan, dtype=float),
        where=delta_a > 0,
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(x, collisions, color="tab:blue", lw=1.0)
    axes[0].set_ylabel("Collisions / step")
    axes[0].set_title("SIDM collisions vs atime")
    axes[0].grid(alpha=0.25)

    positive = np.where(rate_da > 0, rate_da, np.nan)
    axes[1].semilogy(x, positive, color="tab:purple", lw=1.0)
    axes[1].set_ylabel("Collisions / delta-atime")
    axes[1].set_xlabel("atime")
    axes[1].grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_suppressed_vs_collisions(steps: Sequence[StepRecord], out: Path) -> None:
    x = _as_array([st.atime for st in steps])
    collisions = _as_array([st.collisions for st in steps])
    suppressed = _as_array([st.suppressed for st in steps])
    total = collisions + suppressed
    suppressed_frac = np.divide(
        suppressed,
        total,
        out=np.zeros_like(suppressed, dtype=float),
        where=total > 0,
    )

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(x, collisions, color="tab:blue", lw=1.0, label="Collisions")
    ax1.plot(x, suppressed, color="tab:orange", lw=1.0, label="Suppressed")
    ax1.set_xlabel("atime")
    ax1.set_ylabel("Count per step")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, suppressed_frac, color="tab:red", lw=1.0, label="Suppressed fraction")
    ax2.set_ylabel("Suppressed / (Suppressed + Collisions)")
    ax2.set_ylim(0.0, 1.0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Suppressed collisions compared with accepted collisions")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_momentum(records: Sequence[MomentumRecord], out: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    if not records:
        ax1.text(
            0.5,
            0.5,
            "No SIDM momentum warning lines found in log.",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )
        ax1.set_axis_off()
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        return

    ordered = sorted(records, key=lambda r: (r.atime, r.timestamp))
    x = _as_array([r.atime for r in ordered])
    max_ratio = _as_array([r.max_component_ratio for r in ordered])
    l2_ratio = _as_array([r.l2_ratio for r in ordered])
    impacted = _as_array([float(r.impacted) for r in ordered])

    max_ratio = np.where(max_ratio > 0, max_ratio, np.nan)
    l2_ratio = np.where(l2_ratio > 0, l2_ratio, np.nan)

    ax1.semilogy(x, max_ratio, ".", ms=3, color="tab:red", label="max |sum| / abs-sum")
    ax1.semilogy(x, l2_ratio, ".", ms=3, color="tab:blue", label="L2(sum) / L2(abs-sum)")
    ax1.set_xlabel("atime")
    ax1.set_ylabel("Residual ratio")
    ax1.grid(alpha=0.25, which="both")

    ax2 = ax1.twinx()
    ax2.plot(x, impacted, ".", ms=2, color="tab:gray", alpha=0.45, label="impacted particles")
    ax2.set_ylabel("Impacted particles")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("SIDM momentum conservation diagnostic warnings")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SIDM diagnostics from an MP-Gadget SLURM log."
    )
    parser.add_argument("log", type=Path, help="Path to SLURM log file")
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=Path("sidm_log_analysis"),
        help="Output directory for CSV files and plots (default: sidm_log_analysis)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = args.log.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    analyzer = SidmLogAnalyzer()
    analyzer.parse(log_path)

    steps = _valid_steps(analyzer.steps)
    if not steps:
        raise RuntimeError("No valid 'Begin Step' records were parsed from the log.")

    steps.sort(key=lambda s: s.start_ts)
    momentum = sorted(analyzer.momentum, key=lambda r: (r.step, r.timestamp))

    step_csv = outdir / "step_metrics.csv"
    momentum_csv = outdir / "momentum_warnings.csv"
    plot1 = outdir / "1_sidm_cost_comparison.png"
    plot2 = outdir / "2_collision_rate_vs_atime.png"
    plot3 = outdir / "3_suppressed_vs_collisions.png"
    plot4 = outdir / "4_momentum_conservation.png"

    write_step_csv(step_csv, steps)
    write_momentum_csv(momentum_csv, momentum)
    plot_cost_comparison(steps, plot1)
    plot_collision_rate(steps, plot2)
    plot_suppressed_vs_collisions(steps, plot3)
    plot_momentum(momentum, plot4)

    total_wall = sum(st.duration for st in steps)
    total_sidm = sum(st.sidm_time for st in steps)
    total_collisions = sum(st.collisions for st in steps)
    total_suppressed = sum(st.suppressed for st in steps)
    total_prob_reject = sum(st.step1_prob_reject for st in steps)
    total_claimed_partner = sum(st.suppressed_claimed_partner for st in steps)
    total_threshold = sum(st.suppressed_threshold for st in steps)
    total_mutual_skip = sum(st.mutual_skip for st in steps)
    summary_steps = sum(st.summary_present for st in steps)

    print(f"Parsed log: {log_path}")
    print(f"Valid steps: {len(steps)}")
    print(f"Total step walltime (s): {total_wall:.3f}")
    print(f"Total SIDM walltime (s): {total_sidm:.3f}")
    if total_wall > 0:
        print(f"SIDM walltime fraction: {100.0 * total_sidm / total_wall:.2f}%")
    print(f"Total collisions: {int(total_collisions)}")
    print(f"Total suppressed collisions: {int(total_suppressed)}")
    print(f"Steps with SIDM step summary: {summary_steps}")
    if summary_steps > 0:
        print(f"Total prob-reject (step1): {int(total_prob_reject)}")
        print(f"Total suppressed claimed-partner: {int(total_claimed_partner)}")
        print(f"Total suppressed threshold: {int(total_threshold)}")
        print(f"Total mutual-skip: {int(total_mutual_skip)}")
    print(f"Momentum warnings: {len(momentum)}")
    print(f"Wrote CSV: {step_csv}")
    print(f"Wrote CSV: {momentum_csv}")
    print(f"Wrote plot: {plot1}")
    print(f"Wrote plot: {plot2}")
    print(f"Wrote plot: {plot3}")
    print(f"Wrote plot: {plot4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
