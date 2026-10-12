"""Feature engineering blocks and the fold-level preprocessor."""

from .amino_acid import AA_FEATURE_NAMES, aa_features
from .preprocessing import FoldPreprocessor, detect_label_conflicts
from .structural import STRUCT_FEATURE_NAMES, structural_features
from .tda import TDA_FEATURE_NAMES, TDAFeatureExtractor, tda_features

__all__ = [
    "AA_FEATURE_NAMES",
    "STRUCT_FEATURE_NAMES",
    "TDA_FEATURE_NAMES",
    "FoldPreprocessor",
    "TDAFeatureExtractor",
    "aa_features",
    "detect_label_conflicts",
    "structural_features",
    "tda_features",
]
