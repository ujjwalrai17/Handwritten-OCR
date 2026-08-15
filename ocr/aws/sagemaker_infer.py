"""
SageMaker Inference Endpoint + Textract Baseline
Provides:
  1. deploy_endpoint()     — deploy fine-tuned model to SageMaker real-time endpoint
  2. predict_endpoint()    — call the deployed endpoint
  3. textract_baseline()   — run Amazon Textract on an image (zero-setup baseline)
  4. hybrid_predict()      — Textract first; fall back to custom endpoint for low-conf regions

Textract vs Custom Model decision guide:
  ┌─────────────────────────┬──────────────┬──────────────────────────┐
  │ Scenario                │ Use          │ Reason                   │
  ├─────────────────────────┼──────────────┼──────────────────────────┤
  │ Clean printed text      │ Textract     │ Zero latency, no GPU     │
  │ Well-spaced cursive      │ Textract     │ Good enough, cheaper     │
  │ Fast messy cursive       │ Custom model │ TrOCR fine-tuned on IAM  │
  │ Overlapping strokes      │ CRNN+CTC     │ Attention over sequence  │
  │ Mixed document           │ Hybrid       │ Textract + custom fallback│
  └─────────────────────────┴──────────────┴──────────────────────────┘
"""

import base64
import json
import io
from pathlib import Path

import boto3
from config.settings import cfg

SM = cfg.sagemaker


# ── Deploy Endpoint ───────────────────────────────────────────────────────────

def deploy_endpoint(model_s3_uri: str, model_type: str = "trocr") -> str:
    """
    Deploy a trained model to a SageMaker real-time endpoint.

    Args:
        model_s3_uri: S3 URI of the model.tar.gz artifact
        model_type:   "trocr" or "crnn"
    Returns:
        endpoint_name
    """
    import sagemaker
    from sagemaker.huggingface import HuggingFaceModel
    from sagemaker.pytorch import PyTorchModel

    sess = sagemaker.Session(boto3.Session(region_name=SM.region))

    if model_type == "trocr":
        model = HuggingFaceModel(
            model_data=model_s3_uri,
            role=SM.role_arn,
            transformers_version="4.36",
            pytorch_version="2.1",
            py_version="py310",
            env={"HF_TASK": "image-to-text"},
        )
    else:
        model = PyTorchModel(
            model_data=model_s3_uri,
            role=SM.role_arn,
            framework_version="2.1",
            py_version="py310",
            entry_point="ocr/recognition/crnn/engine.py",
        )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=SM.instance_type_infer,
        endpoint_name=SM.endpoint_name,
    )
    print(f"Endpoint deployed: {SM.endpoint_name}")
    return SM.endpoint_name


# ── Call Endpoint ─────────────────────────────────────────────────────────────

def predict_endpoint(image_path: str, endpoint_name: str = None) -> dict:
    """
    Send an image to the SageMaker endpoint and return the OCR result.
    Image is base64-encoded and sent as JSON payload.
    """
    endpoint_name = endpoint_name or SM.endpoint_name
    client = boto3.client("sagemaker-runtime", region_name=SM.region)

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = json.dumps({"image": img_b64})
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload,
    )
    return json.loads(response["Body"].read())


# ── Textract Baseline ─────────────────────────────────────────────────────────

def textract_baseline(image_path: str) -> dict:
    """
    Run Amazon Textract on a local image file.
    Returns dict with 'lines' (list of str) and 'confidence' (mean float).

    Cost: ~$0.0015 per page (DetectDocumentText API).
    Latency: ~1-3s per page.
    Best for: clean/printed text, well-spaced cursive.
    Limitation: struggles with overlapping strokes, very messy cursive.
    """
    client = boto3.client("textract", region_name=SM.region)

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    response = client.detect_document_text(Document={"Bytes": img_bytes})

    lines = []
    confidences = []
    for block in response.get("Blocks", []):
        if block["BlockType"] == "LINE":
            lines.append(block.get("Text", ""))
            confidences.append(block.get("Confidence", 0.0) / 100.0)

    return {
        "lines": lines,
        "confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "raw_blocks": response["Blocks"],
    }


# ── Hybrid Predict ────────────────────────────────────────────────────────────

def hybrid_predict(
    image_path: str,
    textract_threshold: float = 0.85,
    endpoint_name: str = None,
) -> dict:
    """
    Hybrid pipeline:
      1. Run Textract (fast, cheap)
      2. If mean confidence >= textract_threshold → return Textract result
      3. Otherwise → call custom SageMaker endpoint for the full image

    This gives you Textract speed for clean documents and custom model
    accuracy for degraded/cursive regions.

    Args:
        textract_threshold: confidence below which we escalate to custom model
    """
    textract_result = textract_baseline(image_path)
    mean_conf = textract_result["confidence"]

    if mean_conf >= textract_threshold:
        return {
            "source": "textract",
            "lines": textract_result["lines"],
            "confidence": mean_conf,
        }

    # Escalate to custom model
    try:
        custom_result = predict_endpoint(image_path, endpoint_name)
        return {
            "source": "custom_model",
            "lines": custom_result.get("lines", []),
            "confidence": custom_result.get("confidence", 0.0),
            "textract_confidence": mean_conf,
        }
    except Exception as e:
        # Graceful degradation: return Textract result even if low confidence
        return {
            "source": "textract_fallback",
            "lines": textract_result["lines"],
            "confidence": mean_conf,
            "error": str(e),
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    required=True)
    parser.add_argument("--mode",     choices=["textract", "endpoint", "hybrid"],
                        default="hybrid")
    parser.add_argument("--endpoint", default=None)
    args = parser.parse_args()

    if args.mode == "textract":
        result = textract_baseline(args.image)
    elif args.mode == "endpoint":
        result = predict_endpoint(args.image, args.endpoint)
    else:
        result = hybrid_predict(args.image, endpoint_name=args.endpoint)

    print(f"Source: {result.get('source', 'unknown')}")
    print(f"Confidence: {result.get('confidence', 0):.1%}")
    for i, line in enumerate(result.get("lines", []), 1):
        print(f"  [{i:02d}] {line}")
