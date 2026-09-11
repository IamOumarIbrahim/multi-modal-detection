"""Module for filling results tables in README.md from metrics dictionary."""

import difflib
import re
from pathlib import Path
from typing import Union, Optional, Any


def _normalize_key(s: str) -> str:
    s = s.lower().replace("**", "").replace("\\", "")
    return re.sub(r"\s+", " ", s).strip()


def _strip_math(s: str) -> str:
    s = s.lower().replace("**", "").replace("\\", "")
    s = re.sub(r"\s*\([^)]*text[^)]*\)", "", s)
    s = re.sub(r"\s*\([^)]*\$[^)]*\)", "", s)
    s = re.sub(r"\$[^$]*\$", "", s)
    return re.sub(r"\s+", " ", s).strip()


def fill_readme_tables(
    readme_path: Union[str, Path] = "README.md",
    metrics: Optional[dict[str, Any]] = None,
    confirm: bool = False,
    dry_run: bool = False,
) -> str:
    """Fill TBD cells in README.md results tables from metrics dictionary.

    Refuses to write changes to disk unless confirm=True.

    Args:
        readme_path: Path to README.md (or copy).
        metrics: Dictionary mapping (row_key, column_key) or "row_key::column_key"
                 to formatted values.
        confirm: Explicit human confirmation flag required to write to file.
        dry_run: If True, computes and returns diff without writing to disk.

    Returns:
        Unified diff string representing proposed changes.

    Raises:
        PermissionError: If confirm is False and dry_run is False.
        FileNotFoundError: If readme_path does not exist.
    """
    if not confirm and not dry_run:
        raise PermissionError(
            "Refusing to write to README.md without explicit confirmation (confirm=True is required)."
        )

    p = Path(readme_path)
    if not p.exists():
        raise FileNotFoundError(f"README not found at {p}")

    original_content = p.read_text(encoding="utf-8")
    lines = original_content.splitlines(keepends=True)

    metrics_map: dict[str, str] = {}
    if metrics:
        for k, v in metrics.items():
            if isinstance(k, tuple):
                # (row_key, col_key)
                norm_key = f"{str(k[0]).strip()}::{str(k[1]).strip()}"
            else:
                norm_key = str(k).strip()
            metrics_map[norm_key.lower()] = str(v)
            metrics_map[_normalize_key(norm_key)] = str(v)
            metrics_map[_strip_math(norm_key)] = str(v)

    new_lines: list[str] = []
    in_results = False
    current_model = ""
    table_headers: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("### Results"):
            in_results = True
            new_lines.append(line)
            continue
        elif stripped.startswith("## Quick Reproduction"):
            in_results = False
            new_lines.append(line)
            continue

        if in_results and stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]

            # Check if this line is a header
            if any("model" in c.lower() or "post-processing" in c.lower() for c in cells):
                table_headers = cells
                new_lines.append(line)
                continue

            # Check if this line is a separator
            if all(re.match(r"^:?-+:?$", c) for c in cells):
                new_lines.append(line)
                continue

            # It's a data row. Track current model if present
            row_header = cells[0].replace("**", "").strip()
            if row_header in ("YOLO11n", "YOLO26n"):
                current_model = row_header

            sub_header = cells[1].replace("**", "").strip() if len(cells) > 1 else ""

            # Try to match cells with TBD
            new_cells = list(cells)
            row_modified = False

            for col_idx, cell in enumerate(cells):
                if "TBD" in cell:
                    col_name = table_headers[col_idx] if col_idx < len(table_headers) else f"col_{col_idx}"
                    raw_candidates = [
                        f"{current_model}::{sub_header}::{col_name}",
                        f"{sub_header}::{col_name}",
                        f"{row_header}::{col_name}",
                        f"{current_model}::{col_name}",
                    ]

                    replacement = None
                    for cand in raw_candidates:
                        cand_low = cand.lower()
                        cand_norm = _normalize_key(cand)
                        cand_clean = _strip_math(cand)
                        if cand_low in metrics_map:
                            replacement = metrics_map[cand_low]
                            break
                        if cand_norm in metrics_map:
                            replacement = metrics_map[cand_norm]
                            break
                        if cand_clean in metrics_map:
                            replacement = metrics_map[cand_clean]
                            break

                    if replacement is not None:
                        # Preserve bold formatting if present
                        if cell.startswith("**") and cell.endswith("**"):
                            new_cells[col_idx] = f"**{replacement}**"
                        else:
                            new_cells[col_idx] = replacement
                        row_modified = True

            if row_modified:
                new_line = "| " + " | ".join(new_cells) + " |\n"
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    updated_content = "".join(new_lines)
    diff = "".join(
        difflib.unified_diff(
            original_content.splitlines(keepends=True),
            updated_content.splitlines(keepends=True),
            fromfile=str(p),
            tofile=str(p),
        )
    )

    if confirm and not dry_run:
        p.write_text(updated_content, encoding="utf-8")

    return diff
