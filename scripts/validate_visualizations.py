from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


def main() -> None:
    errors = []
    sidecars = list(Path("visualizations/metadata").glob("*.yaml"))
    for sidecar in sidecars:
        metadata = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
        output = Path(metadata["output"])
        if not output.is_file():
            errors.append(f"{metadata['figure_id']}: missing output")
            continue
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        if digest != metadata["output_sha256"]:
            errors.append(f"{metadata['figure_id']}: hash mismatch")
        if metadata["type"] == "empirical" and not metadata.get("source_experiments"):
            errors.append(f"{metadata['figure_id']}: empirical figure lacks experiments")
        if metadata["type"] == "empirical":
            for source in metadata.get("source_artifacts", []):
                source_path = Path(source["path"])
                if not source_path.is_file():
                    errors.append(f"{metadata['figure_id']}: missing source {source_path}")
                elif hashlib.sha256(source_path.read_bytes()).hexdigest() != source["sha256"]:
                    errors.append(f"{metadata['figure_id']}: source hash mismatch {source_path}")
            if not metadata.get("source_artifacts"):
                errors.append(f"{metadata['figure_id']}: empirical figure lacks source hashes")
        invalid_conceptual = (
            metadata["type"] == "conceptual"
            and metadata.get("contains_empirical_data") is not False
        )
        if invalid_conceptual:
            errors.append(f"{metadata['figure_id']}: conceptual evidence flag invalid")
    if errors:
        raise SystemExit("visualization validation failed:\n- " + "\n- ".join(errors))
    print(f"validated {len(sidecars)} visualization provenance records")


if __name__ == "__main__":
    main()
