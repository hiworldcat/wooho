from __future__ import annotations

import argparse
from pathlib import Path

from p1 import geometry as p1_geometry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the P1 weak spatial-geometry checks.")
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--target-root", type=Path, default=None)
    parser.add_argument("--geometry-config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    report = p1_geometry.run_p1_pipeline(
        reference_root=args.reference_root,
        target_root=args.target_root,
        geometry_config_path=args.geometry_config,
        output_root=args.output_root,
    )
    summary = report["summary"]
    print(f"p1 target episodes: {report['dataset']['target_episodes']}")
    print(f"p1 findings: {summary['finding_count']}")
    print(f"p1 comparable episodes: {summary['episodes_with_comparable_pairs']}")
    print(f"p1 panel-proxy episodes: {summary['episodes_with_panel_proxy']}")
    print(f"p1 motion-coupling episodes: {summary.get('episodes_with_motion_coupling', 0)}")


if __name__ == "__main__":
    main()

