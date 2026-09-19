import torch
from torch import nn

from exact_unlearning.contracts import UnlearningContext
from exact_unlearning.models import Classifier, build_model
from exact_unlearning.scrub import (
    ScrubApplicabilityError,
    _forget_objective,
    _has_model_derived_max_step_signal,
    _retain_objective,
    _teacher_student_kl,
    scrub,
)


class DropoutClassifier(Classifier):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(4, 8)
        self.dropout = nn.Dropout(p=0.5)
        self.output = nn.Linear(8, 2)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(torch.relu(self.hidden(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(x))


def test_scrub_kl_is_zero_for_identical_logits() -> None:
    logits = torch.tensor([[1.0, -1.0], [0.5, 0.2]])
    assert abs(float(_teacher_student_kl(logits, logits, 1.0))) < 1e-7


def test_scrub_objectives_match_published_max_min_signs() -> None:
    student = torch.tensor([[1.5, -0.5]], requires_grad=True)
    teacher = torch.tensor([[-0.25, 1.25]])
    labels = torch.tensor([1])

    divergence = _teacher_student_kl(student, teacher, 2.0)
    forget_loss = _forget_objective(student, teacher, 2.0)
    retain_loss = _retain_objective(
        student, teacher, labels, alpha=0.7, gamma=1.3, temperature=2.0
    )

    assert torch.allclose(forget_loss, -divergence)
    assert torch.allclose(
        retain_loss,
        0.7 * divergence + 1.3 * torch.nn.functional.cross_entropy(student, labels),
    )


def test_scrub_rejects_nonpositive_temperature() -> None:
    logits = torch.tensor([[1.0, -1.0]])
    try:
        _teacher_student_kl(logits, logits, 0.0)
    except ValueError as error:
        assert "temperature" in str(error)
    else:
        raise AssertionError("nonpositive temperature must be rejected")


def test_scrub_core_updates_student_without_mutating_teacher() -> None:
    torch.manual_seed(3)
    original = DropoutClassifier()
    before = [parameter.detach().clone() for parameter in original.parameters()]
    context = UnlearningContext(
        original=original,
        x_retain=torch.randn(20, 4),
        y_retain=torch.randint(0, 2, (20,)),
        x_forget=torch.randn(4, 4),
        y_forget=torch.randint(0, 2, (4,)),
        seed=4,
        config={
            "learning_rate": 0.001,
            "batch_size": 8,
            "forget_batch_size": 4,
            "retain_batch_size": 8,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "alpha": 0.5,
            "gamma": 1.0,
            "temperature": 1.0,
            "max_steps": 1,
            "steps": 1,
        },
        device=torch.device("cpu"),
    )
    outcome = scrub(context)
    assert any(
        not torch.equal(left, right)
        for left, right in zip(before, outcome.model.parameters(), strict=True)
    )
    assert all(
        torch.equal(left, right) for left, right in zip(before, original.parameters(), strict=True)
    )
    assert outcome.diagnostics is not None
    trace = outcome.diagnostics["mechanism_trace"]
    assert len(trace) == 1
    assert trace[0]["max_gradient_l2_mean"] > 0
    assert trace[0]["min_gradient_l2_mean"] > 0
    assert set(trace[0]["after_min"]) >= {
        "forget_teacher_student_kl",
        "retain_teacher_student_kl",
        "retain_cross_entropy",
        "retain_objective",
        "forget_accuracy",
        "retain_accuracy",
    }


def test_scrub_fails_closed_when_initial_max_step_has_no_model_signal() -> None:
    torch.manual_seed(3)
    original = build_model("mlp", 4, 8, 2)
    context = UnlearningContext(
        original=original,
        x_retain=torch.randn(20, 4),
        y_retain=torch.randint(0, 2, (20,)),
        x_forget=torch.randn(4, 4),
        y_forget=torch.randint(0, 2, (4,)),
        seed=4,
        config={
            "learning_rate": 0.001,
            "batch_size": 8,
            "optimizer": "adam",
            "alpha": 0.5,
            "gamma": 1.0,
            "temperature": 1.0,
            "max_steps": 1,
            "steps": 2,
        },
        device=torch.device("cpu"),
    )

    try:
        scrub(context)
    except ScrubApplicabilityError as error:
        assert "no model-derived signal" in str(error)
    else:
        raise AssertionError("deterministic identical teacher/student logits must fail closed")


def test_batchnorm_cnn_produces_initial_train_eval_max_step_signal() -> None:
    torch.manual_seed(9)
    model = build_model("small_cnn_bn", (1, 28, 28), 32, 10)
    teacher = model.eval()
    teacher_logits = teacher(torch.randn(8, 1, 28, 28))

    student = build_model("small_cnn_bn", (1, 28, 28), 32, 10)
    student.load_state_dict(model.state_dict())
    student.train()
    probe = torch.randn(8, 1, 28, 28)
    with torch.no_grad():
        teacher_logits = teacher(probe)
        student_logits = student(probe)

    assert _has_model_derived_max_step_signal(student_logits, teacher_logits)


def test_deterministic_cnn_has_no_initial_train_eval_max_step_signal() -> None:
    torch.manual_seed(9)
    model = build_model("small_cnn", (1, 28, 28), 32, 10)
    probe = torch.randn(8, 1, 28, 28)
    model.eval()
    with torch.no_grad():
        teacher_logits = model(probe)
    model.train()
    with torch.no_grad():
        student_logits = model(probe)

    assert not _has_model_derived_max_step_signal(student_logits, teacher_logits)
