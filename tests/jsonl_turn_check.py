import argparse
import json
from collections import defaultdict
from pathlib import Path


def _iter_jsonl_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number} in {path}: {exc}"
                ) from exc


def _collect_dict_fields(value, path: str, fields_by_path: dict[str, set[str]]) -> None:
    if isinstance(value, dict):
        fields_by_path[path].update(str(key) for key in value.keys())
        for key, child in value.items():
            child_path = key if path == "<root>" else f"{path}.{key}"
            _collect_dict_fields(child, child_path, fields_by_path)
        return

    if isinstance(value, list):
        for item in value:
            _collect_dict_fields(item, path, fields_by_path)


def inspect_jsonl(path: Path) -> tuple[int, dict[str, set[str]]]:
    fields_by_path: dict[str, set[str]] = defaultdict(set)
    record_count = 0

    for record in _iter_jsonl_records(path):
        record_count += 1
        _collect_dict_fields(record, "<root>", fields_by_path)

    return record_count, fields_by_path


def format_report(path: Path, record_count: int, fields_by_path: dict[str, set[str]]) -> str:
    lines = [f"Fields found in {path} ({record_count} records):"]

    for field_path in sorted(fields_by_path.keys()):
        field_names = ", ".join(sorted(fields_by_path[field_path]))
        lines.append(f"{field_path}: {{{field_names}}}")

    if len(lines) == 1:
        lines.append("No dict objects found.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a JSONL file and list the fields found for each dict path."
    )
    parser.add_argument("jsonl_path", help="Path to the JSONL file to inspect")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path).expanduser().resolve()
    if not jsonl_path.exists():
        raise FileNotFoundError(f"File not found: {jsonl_path}")
    if not jsonl_path.is_file():
        raise ValueError(f"Expected a file path, got: {jsonl_path}")

    record_count, fields_by_path = inspect_jsonl(jsonl_path)
    print(format_report(jsonl_path, record_count, fields_by_path))


if __name__ == "__main__":
    main()
