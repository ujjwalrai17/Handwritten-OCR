"""
SageMaker Training Pipeline
Launches a SageMaker training job for TrOCR fine-tuning or CRNN+CTC training.

Usage:
    python -m ocr.aws.sagemaker_train --model trocr
    python -m ocr.aws.sagemaker_train --model crnn
    python -m ocr.aws.sagemaker_train --model trocr --hpo  # hyperparameter tuning

Architecture decision:
    Textract baseline  → fast, zero-setup, good for clean/printed text
    TrOCR fine-tuned   → best for cursive, requires GPU training
    CRNN+CTC           → lightweight fallback, fast inference, good for degraded regions
    Ensemble           → Textract for confidence check, custom model for low-conf regions

When to use each:
    - Textract alone:      CER < 5% on your samples, no GPU budget
    - TrOCR fine-tuned:    CER 5-15%, have labeled cursive data (>500 samples)
    - CRNN fallback:       Overlapping/degraded regions where TrOCR confidence < 0.70
    - Full ensemble:       Production system, maximize accuracy across all quality levels
"""

import json
import boto3
from config.settings import cfg

SM = cfg.sagemaker


def launch_trocr_training(
    s3_train_data: str = None,
    s3_val_data: str = None,
    hyperparameters: dict = None,
) -> str:
    """
    Launch SageMaker training job for TrOCR fine-tuning.
    Returns the training job name.
    """
    import sagemaker
    from sagemaker.huggingface import HuggingFace

    sess = sagemaker.Session(boto3.Session(region_name=SM.region))

    hp = {
        "epochs":        str(cfg.training.num_epochs),
        "lr":            str(cfg.training.learning_rate),
        "batch":         str(cfg.training.train_batch_size),
        "warmup_steps":  str(cfg.training.warmup_steps),
        "weight_decay":  str(cfg.training.weight_decay),
        "max_target_len": str(cfg.training.max_target_length),
        "model_name":    cfg.model.name,
    }
    if hyperparameters:
        hp.update({k: str(v) for k, v in hyperparameters.items()})

    estimator = HuggingFace(
        entry_point="train.py",
        source_dir=".",
        role=SM.role_arn,
        instance_type=SM.instance_type_train,
        instance_count=SM.instance_count,
        volume_size=SM.volume_size_gb,
        max_run=SM.max_runtime_sec,
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        hyperparameters=hp,
        metric_definitions=[
            {"Name": "train:loss",  "Regex": r"Loss (\S+)"},
            {"Name": "val:cer",     "Regex": r"CER (\S+)"},
            {"Name": "val:wer",     "Regex": r"WER (\S+)"},
        ],
        checkpoint_s3_uri=f"{SM.s3_bucket}/checkpoints/trocr",
    )

    data_channels = {}
    if s3_train_data:
        data_channels["train"] = s3_train_data
    if s3_val_data:
        data_channels["validation"] = s3_val_data

    estimator.fit(data_channels or None, wait=False)
    job_name = estimator.latest_training_job.name
    print(f"TrOCR training job launched: {job_name}")
    return job_name


def launch_crnn_training(
    s3_train_data: str = None,
    hyperparameters: dict = None,
) -> str:
    """Launch SageMaker training job for CRNN+CTC."""
    import sagemaker
    from sagemaker.pytorch import PyTorch

    sess = sagemaker.Session(boto3.Session(region_name=SM.region))

    hp = {
        "epochs": "30",
        "lr":     "1e-3",
        "batch":  "16",
    }
    if hyperparameters:
        hp.update({k: str(v) for k, v in hyperparameters.items()})

    estimator = PyTorch(
        entry_point="train_crnn.py",
        source_dir=".",
        role=SM.role_arn,
        instance_type=SM.instance_type_train,
        instance_count=1,
        volume_size=SM.volume_size_gb,
        framework_version="2.1",
        py_version="py310",
        hyperparameters=hp,
        metric_definitions=[
            {"Name": "val:cer",  "Regex": r"CER (\S+)"},
            {"Name": "val:wer",  "Regex": r"WER (\S+)"},
            {"Name": "overlap:cer", "Regex": r"Overlap-subset CER (\S+)"},
        ],
        checkpoint_s3_uri=f"{SM.s3_bucket}/checkpoints/crnn",
    )

    estimator.fit({"train": s3_train_data} if s3_train_data else None, wait=False)
    job_name = estimator.latest_training_job.name
    print(f"CRNN training job launched: {job_name}")
    return job_name


def launch_hpo(model: str = "trocr") -> str:
    """
    Hyperparameter optimization job focused on improving recognition of
    ambiguous/overlapping strokes.

    Key HPO targets:
      - learning_rate:    log-uniform [1e-5, 1e-4]  (TrOCR) / [1e-4, 1e-2] (CRNN)
      - warmup_steps:     integer [100, 1000]
      - batch_size:       categorical [4, 8, 16]
      - elastic_alpha:    continuous [20, 60]   (augmentation intensity)
      - elastic_sigma:    continuous [2, 8]
      - line_bleed_prob:  continuous [0.2, 0.6]
    """
    import sagemaker
    from sagemaker.tuner import (
        HyperparameterTuner, ContinuousParameter, IntegerParameter, CategoricalParameter
    )
    from sagemaker.huggingface import HuggingFace
    from sagemaker.pytorch import PyTorch

    sess = sagemaker.Session(boto3.Session(region_name=SM.region))

    if model == "trocr":
        estimator = HuggingFace(
            entry_point="train.py",
            source_dir=".",
            role=SM.role_arn,
            instance_type=SM.instance_type_train,
            instance_count=1,
            transformers_version="4.36",
            pytorch_version="2.1",
            py_version="py310",
            hyperparameters={"epochs": "5", "model_name": cfg.model.name},
        )
        hp_ranges = {
            "lr":           ContinuousParameter(1e-5, 1e-4, scaling_type="Logarithmic"),
            "warmup_steps": IntegerParameter(100, 1000),
            "batch":        CategoricalParameter(["4", "8"]),
        }
    else:  # crnn
        estimator = PyTorch(
            entry_point="train_crnn.py",
            source_dir=".",
            role=SM.role_arn,
            instance_type=SM.instance_type_train,
            instance_count=1,
            framework_version="2.1",
            py_version="py310",
            hyperparameters={"epochs": "15"},
        )
        hp_ranges = {
            "lr":              ContinuousParameter(1e-4, 1e-2, scaling_type="Logarithmic"),
            "batch":           CategoricalParameter(["8", "16", "32"]),
            "elastic_alpha":   ContinuousParameter(20.0, 60.0),
            "elastic_sigma":   ContinuousParameter(2.0, 8.0),
            "line_bleed_prob": ContinuousParameter(0.2, 0.6),
        }

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name="val:cer",
        objective_type="Minimize",
        hyperparameter_ranges=hp_ranges,
        max_jobs=12,
        max_parallel_jobs=2,
        strategy="Bayesian",
    )
    tuner.fit(wait=False)
    job_name = tuner.latest_tuning_job.name
    print(f"HPO job launched: {job_name}")
    return job_name


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["trocr", "crnn"], default="trocr")
    parser.add_argument("--hpo",   action="store_true")
    parser.add_argument("--s3-train", default=None)
    parser.add_argument("--s3-val",   default=None)
    args = parser.parse_args()

    if args.hpo:
        launch_hpo(args.model)
    elif args.model == "trocr":
        launch_trocr_training(args.s3_train, args.s3_val)
    else:
        launch_crnn_training(args.s3_train)
