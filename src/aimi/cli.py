from __future__ import annotations

import argparse

from aimi.generator import GeneratorConfig, SyntheticBatchGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--batches", type=int, default=200)
    gen.add_argument("--output", type=str, default="data/synthetic_batches.csv")
    gen.add_argument("--profile-output", type=str, default="data/synthetic_profiles.csv")
    args = parser.parse_args()

    if args.cmd == "generate":
        bdf, pdf = SyntheticBatchGenerator(GeneratorConfig(n_batches=args.batches)).generate()
        bdf.to_csv(args.output, index=False)
        pdf.to_csv(args.profile_output, index=False)
        print(f"Wrote {args.output} and {args.profile_output}")


if __name__ == "__main__":
    main()
