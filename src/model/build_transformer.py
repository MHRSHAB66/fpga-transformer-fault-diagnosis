from __future__ import annotations

import json

from src.models.transformer_baseline import TransformerBaseline


def build_model() -> TransformerBaseline:
    return TransformerBaseline()


def main():
    model = build_model()
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    summary = {
        "model_name": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }

    print(model)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
