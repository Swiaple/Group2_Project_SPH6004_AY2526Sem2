from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from utils.cma_dataset import CmaArtifacts, CmaTensorDataset, build_cma_data_bundle


@dataclass
class MultimodalArtifacts:
    cma_artifacts: CmaArtifacts
    use_text: bool
    use_time: bool
    use_static: bool


def _apply_modality_toggles(dataset: CmaTensorDataset, use_text: bool, use_time: bool, use_static: bool) -> None:
    if not use_text:
        dataset.input_ids.zero_()
        dataset.attention_mask.zero_()
    if not use_time:
        dataset.time_tensor.zero_()
    if not use_static:
        dataset.static_tensor.zero_()


def build_multimodal_data_bundle(
    debug_max_stays: int = 0,
    tokenizer_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    max_text_len: int = 128,
    time_window_hours: int = 24,
    use_text: bool = True,
    use_time: bool = True,
    use_static: bool = True,
) -> Dict[str, object]:
    """
    Appendix-A aligned multimodal data builder.
    Reuses the CMA data contract and supports modality toggles for ablations.
    """
    bundle = build_cma_data_bundle(
        debug_max_stays=debug_max_stays,
        tokenizer_name=tokenizer_name,
        max_text_len=max_text_len,
        time_window_hours=time_window_hours,
    )

    for split_name in ("train_dataset", "val_dataset", "test_dataset"):
        _apply_modality_toggles(
            dataset=bundle[split_name],
            use_text=use_text,
            use_time=use_time,
            use_static=use_static,
        )

    bundle["artifacts"] = MultimodalArtifacts(
        cma_artifacts=bundle["artifacts"],
        use_text=bool(use_text),
        use_time=bool(use_time),
        use_static=bool(use_static),
    )
    bundle["modality_toggles"] = {
        "use_text": bool(use_text),
        "use_time": bool(use_time),
        "use_static": bool(use_static),
    }
    return bundle
