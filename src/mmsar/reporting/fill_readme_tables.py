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
    table_headers: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## Results") or stripped.startswith("### Results"):
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
            header_keywords = ("configuration", "method", "model", "modality", "post-processing")
            if any(any(k in cell.lower() for k in header_keywords) for cell in cells):
                table_headers = cells
                new_lines.append(line)
                continue

            # Check if this line is a separator
            if all(re.match(r"^:?-+:?$", c) for c in cells):
                new_lines.append(line)
                continue

            # It's a data row
            row_header = cells[0].replace("**", "").strip()
            clean_row = re.sub(r"\(.*?\)", "", row_header).strip()

            model_prefix = ""
            if "yolo11" in clean_row.lower():
                model_prefix = "yolo11n"
            elif "yolo26" in clean_row.lower():
                model_prefix = "yolo26n"

            modality_str = ""
            if "rgb" in clean_row.lower() or (len(cells) > 1 and "rgb" in cells[1].lower()):
                modality_str = "rgb stream"
            elif "thermal" in clean_row.lower() or (len(cells) > 1 and "thermal" in cells[1].lower()):
                modality_str = "thermal stream"
            elif "fusion" in clean_row.lower():
                modality_str = "late fusion gate"

            env_str = ""
            for c in cells:
                c_low = c.lower()
                if "desert" in c_low:
                    env_str = "desert"
                    break
                elif "forest" in c_low:
                    env_str = "forest"
                    break

            method_name = ""
            method_prefix = ""
            if ":" in clean_row:
                parts = clean_row.split(":", 1)
                method_prefix = parts[0].strip()
                method_name = parts[1].strip()
            elif any(k in clean_row.lower() for k in ("baseline", "moving average", "median", "consensus", "mamba")):
                method_name = clean_row

            # Try to match cells with TBD
            new_cells = list(cells)
            row_modified = False

            for col_idx, cell in enumerate(cells):
                if "TBD" in cell:
                    col_name = table_headers[col_idx] if col_idx < len(table_headers) else f"col_{col_idx}"
                    clean_col = re.sub(r"\(.*?\)", "", col_name).replace("$", "").replace("\\", "").strip()

                    raw_candidates = [
                        f"{row_header}::{col_name}",
                        f"{clean_row}::{clean_col}",
                        f"{clean_row}::{col_name}",
                    ]

                    if model_prefix:
                        raw_candidates.extend([
                            f"{model_prefix}::{col_name}",
                            f"{model_prefix}::{clean_col}",
                        ])
                        if modality_str:
                            raw_candidates.extend([
                                f"{model_prefix}::{modality_str}::{col_name}",
                                f"{model_prefix}::{modality_str}::{clean_col}",
                                f"{model_prefix}::{modality_str} ($c_{{\\text{{{modality_str[:3]}}}}})::{col_name}",
                                f"{model_prefix}::{modality_str} ($c_{{\\text{{{modality_str[:3]}}}}})::{clean_col}",
                            ])
                        if env_str:
                            raw_candidates.extend([
                                f"{model_prefix}::{env_str}::{col_name}",
                                f"{model_prefix}::{env_str}::{clean_col}",
                            ])
                            if modality_str:
                                raw_candidates.extend([
                                    f"{model_prefix}::{modality_str}::{env_str}::{col_name}",
                                    f"{model_prefix}::{modality_str}::{env_str}::{clean_col}",
                                ])

                    if method_name:
                        raw_candidates.extend([
                            f"{method_name}::{col_name}",
                            f"{method_name}::{clean_col}",
                            f"{method_name.split()[0]}::{col_name}",
                            f"{method_name.split()[0]}::{clean_col}",
                            f"yolo11n::{method_name.lower()}::{col_name}",
                            f"yolo11n::{method_name.lower()}::{clean_col}",
                            f"yolo11n::{method_name.split()[0].lower()}::{col_name}",
                            f"yolo11n::{method_name.split()[0].lower()}::{clean_col}",
                        ])
                        if "baseline" in method_name.lower():
                            raw_candidates.extend([
                                f"baseline::{col_name}",
                                f"baseline::{clean_col}",
                                f"baseline::desert false alarms",
                                f"yolo11n::baseline::{col_name}",
                                f"yolo11n::baseline::{clean_col}",
                            ])

                    if method_prefix:
                        raw_candidates.extend([
                            f"{method_prefix}::{col_name}",
                            f"{method_prefix}::{clean_col}",
                        ])

                    # Expand for precision / recall aliases
                    expanded = []
                    for cand in raw_candidates:
                        expanded.append(cand)
                        cand_l = cand.lower()
                        if "precision" in cand_l and "frame precision" not in cand_l:
                            expanded.append(re.sub(r"::precision", "::frame precision", cand, flags=re.IGNORECASE))
                        elif "recall" in cand_l and "frame recall" not in cand_l:
                            expanded.append(re.sub(r"::recall", "::frame recall", cand, flags=re.IGNORECASE))

                    replacement = None
                    for cand in expanded:
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
                # Maintain original line ending (CRLF or LF)
                line_ending = "\r\n" if line.endswith("\r\n") else "\n"
                new_line = "| " + " | ".join(new_cells) + " |" + line_ending
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
