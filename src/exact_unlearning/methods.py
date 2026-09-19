from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .contracts import UnlearningContext
from .unlearning import UnlearningResult, fine_tune, naive_gradient_ascent, neggrad_plus


@dataclass(frozen=True)
class MethodDescriptor:
    name: str
    simple_baseline: bool
    literature_reference: str | None
    implementation_provenance: str
    implementation_notes: str
    runner: Callable[[UnlearningContext], UnlearningResult]


def _fine_tune(context: UnlearningContext) -> UnlearningResult:
    return fine_tune(
        context.original,
        context.x_retain,
        context.y_retain,
        seed=context.seed,
        epochs=int(context.config["epochs"]),
        batch_size=int(context.config["batch_size"]),
        learning_rate=float(context.config["learning_rate"]),
        device=context.device,
    )


def _naive_gradient_ascent(context: UnlearningContext) -> UnlearningResult:
    return naive_gradient_ascent(
        context.original,
        context.x_retain,
        context.y_retain,
        context.x_forget,
        context.y_forget,
        seed=context.seed,
        epochs=int(context.config["epochs"]),
        batch_size=int(context.config["batch_size"]),
        learning_rate=float(context.config["learning_rate"]),
        retain_weight=float(context.config["retain_weight"]),
        forget_weight=float(context.config["forget_weight"]),
        device=context.device,
    )


def _scrub(context: UnlearningContext) -> UnlearningResult:
    from .scrub import scrub

    return scrub(context)


def _neggrad_plus(context: UnlearningContext) -> UnlearningResult:
    return neggrad_plus(
        context.original,
        context.x_retain,
        context.y_retain,
        context.x_forget,
        context.y_forget,
        seed=context.seed,
        epochs=int(context.config["epochs"]),
        batch_size=int(context.config["batch_size"]),
        learning_rate=float(context.config["learning_rate"]),
        beta=float(context.config["beta"]),
        weight_decay=float(context.config.get("weight_decay", 0.0)),
        momentum=float(context.config.get("momentum", 0.9)),
        device=context.device,
    )


METHODS = {
    "fine_tune": MethodDescriptor(
        "fine_tune", True, None, "local baseline", "Retained-data fine-tuning.", _fine_tune
    ),
    "naive_gradient_ascent": MethodDescriptor(
        "naive_gradient_ascent",
        True,
        None,
        "local baseline",
        "Naive retain-descent/forget-ascent diagnostic; not a contemporary SOTA method.",
        _naive_gradient_ascent,
    ),
    "neggrad_plus": MethodDescriptor(
        "neggrad_plus",
        False,
        "Kurmanji et al., NeurIPS 2023, DOI 10.52202/075280-0095, Equation 4",
        "Declared clean-room implementation from the published NegGrad+ objective; no official "
        "repository source is included.",
        "Literature baseline with beta-weighted retain descent and forget ascent. It remains "
        "ineligible for comparative experiments pending a fixed tuning budget and reproduction.",
        _neggrad_plus,
    ),
    "scrub": MethodDescriptor(
        "scrub",
        False,
        "Kurmanji et al., NeurIPS 2023, DOI 10.52202/075280-0095",
        "Declared clean-room implementation from paper Equations 1--3 and Algorithms 1--3; "
        "the authors' repository and license were inspected, but behavioral reproduction "
        "remains pending.",
        "Core alternating max/min objective. Pilot omits privacy rewinding and uses declared "
        "Fashion-MNIST hyperparameters. The authors' repository carries a non-commercial academic "
        "software license; no third-party source is declared as included, and human provenance "
        "review remains required before public method release.",
        _scrub,
    ),
}


def run_unlearning_method(
    name: str, context: UnlearningContext
) -> tuple[MethodDescriptor, UnlearningResult]:
    try:
        descriptor = METHODS[name]
    except KeyError as error:
        raise ValueError(f"unregistered unlearning method: {name}") from error
    return descriptor, descriptor.runner(context)
