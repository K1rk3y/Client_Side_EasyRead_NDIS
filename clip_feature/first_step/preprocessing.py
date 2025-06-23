import re
from pathlib import Path


def standardize_filename(path: Path, pad: int = 2) -> Path:
    """
    Given a Path like "Advocate group presentation 01.png", returns a new Path
    like "advocate_group_presentation_01.png".
    """
    stem, ext = path.stem, path.suffix
    # 1) Remove any parenthetical content "(...)" and trailing tags like "- Edited"
    #    and unify separators to spaces
    cleaned = re.sub(r"\(.*?\)", "", stem)  # drop (... )
    cleaned = re.sub(r"[-–—]+", " ", cleaned)  # hyphens/dashes → spaces
    # 2) Extract trailing number if present
    m = re.search(r"(.*?)(\d+)\s*$", cleaned)
    if m:
        name_part, num_part = m.group(1), m.group(2)
    else:
        name_part, num_part = cleaned, ""
    # 3) Turn any non-alphanumeric chars into spaces, split and re-join on underscores
    words = re.findall(r"[A-Za-z0-9]+", name_part)
    base = "_".join(w.lower() for w in words if not w.isdigit())
    # 4) Zero-pad the numeric part (or leave blank if none)
    if num_part:
        num = num_part.zfill(pad)
        new_stem = f"{base}_{num}"
    else:
        new_stem = base
    return path.with_name(new_stem + ext.lower())


def batch_rename(folder: Path, extensions=("*.png",)):
    """
    Rename all files in `folder` matching the given glob extensions.
    """
    for ext in extensions:
        for path in folder.glob(ext):
            new_path = standardize_filename(path)
            if new_path != path:
                print(f"Renaming:\n  {path.name}  →  {new_path.name}")
                path.rename(new_path)


if __name__ == "__main__":
    from pathlib import Path

    batch_rename(Path("images/pilot"), extensions=["*.png"])
