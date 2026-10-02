#!/usr/bin/env python3
"""
ec2_tags_from_ips.py — Look up EC2 instance tags by IP address.

Usage:
    # From a file (one IP per line)
    python3 scripts/ec2_tags_from_ips.py --input ips.txt

    # From stdin
    echo -e "10.0.1.5\n10.0.2.10" | python3 scripts/ec2_tags_from_ips.py

    # Inline IPs
    python3 scripts/ec2_tags_from_ips.py --ips 10.0.1.5 10.0.2.10

    # Filter to specific tag keys
    python3 scripts/ec2_tags_from_ips.py --input ips.txt --tags Name Environment

    # JSON output
    python3 scripts/ec2_tags_from_ips.py --input ips.txt --format json

Options:
    --input FILE      File with one IP per line (ignores blank lines and # comments)
    --ips IP [IP...]  IPs passed directly on the command line
    --tags KEY [...]  Only show these tag keys (default: all)
    --format          Output format: table (default) | json | csv
    --region REGION   AWS region (falls back to AWS_DEFAULT_REGION / boto3 default)
    --profile NAME    AWS profile name
"""

import argparse
import csv
import json
import sys
from collections import defaultdict

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def parse_args():
    p = argparse.ArgumentParser(description="Find EC2 tags from a list of IP addresses.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--input", metavar="FILE", help="File with one IP per line")
    src.add_argument("--ips", nargs="+", metavar="IP", help="IPs on the command line")
    p.add_argument("--tags", nargs="+", metavar="KEY", help="Tag keys to display (default: all)")
    p.add_argument("--format", choices=["table", "json", "csv"], default="table")
    p.add_argument("--region", help="AWS region")
    p.add_argument("--profile", help="AWS named profile")
    return p.parse_args()


def load_ips(args) -> list[str]:
    if args.ips:
        return [ip.strip() for ip in args.ips if ip.strip()]
    if args.input:
        with open(args.input) as f:
            lines = f.readlines()
    else:
        if sys.stdin.isatty():
            print("No IPs provided. Use --input, --ips, or pipe IPs via stdin.", file=sys.stderr)
            sys.exit(1)
        lines = sys.stdin.readlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def tags_to_dict(tag_list: list) -> dict:
    return {t["Key"]: t["Value"] for t in (tag_list or [])}


def lookup_instances(ec2, ips: list[str]) -> list[dict]:
    """
    Query EC2 for instances whose private OR public IP matches any of the given IPs.
    Returns a flat list of result dicts.
    """
    results = []

    # Build an IP → original-query map so we can report not-found IPs later.
    remaining = set(ips)

    filters_private = [{"Name": "private-ip-address", "Values": ips}]
    filters_public = [{"Name": "ip-address", "Values": ips}]

    seen_instance_ids = set()

    for filters in (filters_private, filters_public):
        paginator = ec2.get_paginator("describe_instances")
        for page in paginator.paginate(Filters=filters):
            for reservation in page["Reservations"]:
                for inst in reservation["Instances"]:
                    iid = inst["InstanceId"]
                    if iid in seen_instance_ids:
                        continue
                    seen_instance_ids.add(iid)

                    private_ip = inst.get("PrivateIpAddress", "")
                    public_ip = inst.get("PublicIpAddress", "")
                    matched_ip = private_ip if private_ip in remaining else public_ip

                    remaining.discard(private_ip)
                    remaining.discard(public_ip)

                    results.append(
                        {
                            "queried_ip": matched_ip,
                            "instance_id": iid,
                            "private_ip": private_ip,
                            "public_ip": public_ip,
                            "state": inst.get("State", {}).get("Name", ""),
                            "instance_type": inst.get("InstanceType", ""),
                            "tags": tags_to_dict(inst.get("Tags", [])),
                        }
                    )

    # Add not-found entries
    for ip in remaining:
        results.append(
            {
                "queried_ip": ip,
                "instance_id": "NOT FOUND",
                "private_ip": "",
                "public_ip": "",
                "state": "",
                "instance_type": "",
                "tags": {},
            }
        )

    return sorted(results, key=lambda r: r["queried_ip"])


def collect_all_tag_keys(results: list[dict]) -> list[str]:
    keys = set()
    for r in results:
        keys.update(r["tags"].keys())
    return sorted(keys)


def output_table(results: list[dict], tag_keys: list[str]):
    fixed_cols = ["queried_ip", "instance_id", "state", "instance_type", "private_ip", "public_ip"]
    all_cols = fixed_cols + tag_keys

    # Determine column widths
    widths = {col: len(col) for col in all_cols}
    for r in results:
        widths["queried_ip"] = max(widths["queried_ip"], len(r["queried_ip"]))
        widths["instance_id"] = max(widths["instance_id"], len(r["instance_id"]))
        widths["state"] = max(widths["state"], len(r["state"]))
        widths["instance_type"] = max(widths["instance_type"], len(r["instance_type"]))
        widths["private_ip"] = max(widths["private_ip"], len(r["private_ip"]))
        widths["public_ip"] = max(widths["public_ip"], len(r["public_ip"]))
        for k in tag_keys:
            widths[k] = max(widths[k], len(r["tags"].get(k, "")))

    header = "  ".join(col.upper().ljust(widths[col]) for col in all_cols)
    sep = "  ".join("-" * widths[col] for col in all_cols)
    print(header)
    print(sep)
    for r in results:
        row_vals = [
            r["queried_ip"],
            r["instance_id"],
            r["state"],
            r["instance_type"],
            r["private_ip"],
            r["public_ip"],
        ] + [r["tags"].get(k, "") for k in tag_keys]
        print("  ".join(v.ljust(widths[col]) for v, col in zip(row_vals, all_cols)))


def output_json(results: list[dict]):
    print(json.dumps(results, indent=2))


def output_csv(results: list[dict], tag_keys: list[str]):
    fixed_cols = ["queried_ip", "instance_id", "state", "instance_type", "private_ip", "public_ip"]
    writer = csv.writer(sys.stdout)
    writer.writerow(fixed_cols + tag_keys)
    for r in results:
        row = [
            r["queried_ip"],
            r["instance_id"],
            r["state"],
            r["instance_type"],
            r["private_ip"],
            r["public_ip"],
        ] + [r["tags"].get(k, "") for k in tag_keys]
        writer.writerow(row)


def main():
    args = parse_args()
    ips = load_ips(args)

    if not ips:
        print("No IPs to look up.", file=sys.stderr)
        sys.exit(1)

    print(f"Looking up {len(ips)} IP(s)…", file=sys.stderr)

    session_kwargs = {}
    if args.region:
        session_kwargs["region_name"] = args.region
    if args.profile:
        session_kwargs["profile_name"] = args.profile

    session = boto3.Session(**session_kwargs)
    ec2 = session.client("ec2")

    try:
        results = lookup_instances(ec2, ips)
    except (BotoCoreError, ClientError) as exc:
        print(f"AWS error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Decide which tag keys to display
    if args.tags:
        tag_keys = args.tags
    else:
        tag_keys = collect_all_tag_keys(results)

    if args.format == "json":
        output_json(results)
    elif args.format == "csv":
        output_csv(results, tag_keys)
    else:
        output_table(results, tag_keys)

    not_found = [r for r in results if r["instance_id"] == "NOT FOUND"]
    if not_found:
        print(
            f"\n{len(not_found)} IP(s) not found: {', '.join(r['queried_ip'] for r in not_found)}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
