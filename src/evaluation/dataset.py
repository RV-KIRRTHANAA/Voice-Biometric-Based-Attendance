import csv
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

VALID_SAMPLE_TYPES = {'reference', 'genuine', 'impostor', 'replay', 'synthetic'}

@dataclass
class EvalSample:
    sample_id: str
    speaker_id: str
    sample_type: str  # 'reference', 'genuine', 'impostor', 'replay', 'synthetic'
    reference_id: Optional[str]
    condition: str
    path: str
    absolute_path: str

class DatasetValidator:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.manifest_path = self.data_dir / "manifest.csv"
        
    def load_and_validate(self) -> List[EvalSample]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")
            
        samples: List[EvalSample] = []
        sample_ids = set()
        reference_sample_ids = set()
        reference_group_ids = set()
        speakers_with_references = set()
        
        with open(self.manifest_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Ensure required columns exist
            required_cols = {'sample_id', 'speaker_id', 'sample_type', 'reference_id', 'condition', 'path'}
            if not required_cols.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"Manifest missing required columns. Found: {reader.fieldnames}")
                
            for row_idx, row in enumerate(reader, start=2): # +1 for header, +1 for 1-based index
                sample_id = row['sample_id'].strip()
                
                if not sample_id:
                    raise ValueError(f"Row {row_idx}: sample_id is empty.")
                if sample_id in sample_ids:
                    raise ValueError(f"Row {row_idx}: Duplicate sample_id '{sample_id}'.")
                    
                sample_ids.add(sample_id)
                
                # Resolve path securely within data_dir
                rel_path = row['path'].strip()
                abs_path = (self.data_dir / rel_path).resolve()
                
                # Security check: must remain inside data_dir
                try:
                    abs_path.relative_to(self.data_dir.resolve())
                except ValueError:
                    raise ValueError(f"Row {row_idx}: Path '{rel_path}' attempts to escape data directory.")
                
                if not abs_path.exists():
                    raise FileNotFoundError(f"Row {row_idx}: Audio file not found at '{abs_path}'.")
                    
                speaker_id = row['speaker_id'].strip()
                sample_type = row['sample_type'].strip().lower()
                ref_id = row['reference_id'].strip()
                
                if sample_type == 'reference':
                    reference_sample_ids.add(sample_id)
                    speakers_with_references.add(speaker_id)
                    if ref_id:
                        reference_group_ids.add(ref_id)
                        
                samples.append(EvalSample(
                    sample_id=sample_id,
                    speaker_id=speaker_id,
                    sample_type=sample_type,
                    reference_id=ref_id if ref_id else None,
                    condition=row['condition'].strip(),
                    path=rel_path,
                    absolute_path=str(abs_path)
                ))
                
        # Second pass: validate references on non-reference attempt rows.
        # Supports:
        # 1. A single reference sample_id (e.g. 'r1')
        # 2. Semicolon-separated reference sample_ids (e.g. 'r1;r2;r3')
        # 3. A reference group id shared by reference rows (e.g. 'ref_s1')
        # 4. A speaker_id that has reference rows in the manifest (e.g. 's1')
        valid_ref_targets = sample_ids | reference_group_ids | speakers_with_references
        for sample in samples:
            if sample.sample_type == 'reference':
                continue
            if sample.reference_id:
                tokens = [t.strip() for t in sample.reference_id.split(';') if t.strip()]
                if not tokens:
                    raise ValueError(f"Sample '{sample.sample_id}' refers to unknown reference_id '{sample.reference_id}'")
                for token in tokens:
                    if token not in valid_ref_targets:
                        raise ValueError(f"Sample '{sample.sample_id}' refers to unknown reference_id '{sample.reference_id}'")
                
        return samples
