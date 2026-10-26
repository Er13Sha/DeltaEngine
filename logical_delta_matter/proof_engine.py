from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


SOUND = "SOUND"
UNKNOWN = "UNKNOWN"
REJECTED_CERTIFICATE = "REJECTED_CERTIFICATE"


@dataclass(frozen=True, slots=True)
class ProofObligation:
    obligation_id: str
    domain: str
    predicate: str
    total: str
    exclusive: str
    termination: str

    def to_dict(self) -> dict[str, str]:
        return {
            "obligation_id": self.obligation_id,
            "domain": self.domain,
            "predicate": self.predicate,
            "total": self.total,
            "exclusive": self.exclusive,
            "termination": self.termination,
        }


@dataclass(frozen=True, slots=True)
class ProofCertificate:
    certificate_id: str
    status: str
    engine: str
    obligations: tuple[ProofObligation, ...]
    artifacts: tuple[dict[str, Any], ...]
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "status": self.status,
            "engine": self.engine,
            "reason": self.reason,
            "obligations": [item.to_dict() for item in self.obligations],
            "artifacts": list(self.artifacts),
        }


def reject_unsafe_proof_text(text: str) -> str | None:
    forbidden = ("Admitted.", "admit", "sorry", "axiom ", "Axiom ", "unsafe")
    lowered = text.lower()
    for token in forbidden:
        if token.lower() in lowered:
            return f"forbidden proof token: {token}"
    return None


def ensure_no_float(value: Any, path: str = "$") -> None:
    if type(value) is float:
        raise TypeError(f"Float is forbidden in trusted path at {path}.")
    if isinstance(value, dict):
        for key, item in value.items():
            ensure_no_float(key, f"{path}.key")
            ensure_no_float(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            ensure_no_float(item, f"{path}[{index}]")


class StrictProofEngine:
    """Sound-or-Unknown certificate builder for finite delta decisions."""

    def certificate_for_decision(self, decision: Any, artifacts: Iterable[dict[str, Any]] = ()) -> ProofCertificate:
        decision_dict = decision.to_dict() if hasattr(decision, "to_dict") else dict(decision)
        try:
            ensure_no_float(decision_dict)
        except TypeError as exc:
            return self._rejected("trusted_path.float", str(exc))

        for artifact in artifacts:
            text = artifact.get("text", "") if isinstance(artifact, dict) else ""
            if text:
                reason = reject_unsafe_proof_text(text)
                if reason:
                    return self._rejected("trusted_path.proof_text", reason)

        trace = decision_dict.get("trace", [])
        if not trace:
            return self._unknown("trusted_path.empty_trace", "Decision has no delta trace.")

        obligations: list[ProofObligation] = []
        for index, step in enumerate(trace):
            if type(step.get("value")) is not bool:
                return self._rejected("trusted_path.non_boolean_delta", f"Delta {step.get('delta_id')} is not boolean.")
            branches = step.get("branches", {})
            if set(branches) != {"true", "false"}:
                return self._rejected("trusted_path.non_binary_delta", f"Delta {step.get('delta_id')} lacks true/false branches.")
            if branches["true"] == branches["false"]:
                return self._rejected("trusted_path.non_exclusive_delta", f"Delta {step.get('delta_id')} branches collapse.")
            obligations.append(
                ProofObligation(
                    obligation_id=f"po.{index}.{step['delta_id']}",
                    domain=decision_dict.get("domain", "UnknownDomain"),
                    predicate=step["predicate"],
                    total="predicate is defined by typed finite input fields",
                    exclusive="boolean branch P / not P",
                    termination="single predicate evaluation in finite trace",
                )
            )

        return ProofCertificate(
            certificate_id=f"cert.{decision_dict.get('domain', 'domain')}.{decision_dict.get('leaf', 'leaf')}",
            status=SOUND,
            engine="strict-proof-engine@0.1",
            obligations=tuple(obligations),
            artifacts=tuple(artifacts),
            reason="finite boolean trace validated",
        )

    def partition_certificate(self, delta_ids: Iterable[str]) -> ProofCertificate:
        ids = tuple(delta_ids)
        if not ids:
            return self._unknown("partition.empty", "Cannot certify an empty partition.")
        if len(set(ids)) != len(ids):
            return self._rejected("partition.duplicate_delta", "Delta ids must be unique.")
        obligations = tuple(
            ProofObligation(
                obligation_id=f"partition.{index}.{delta_id}",
                domain="BooleanDeltaPartition",
                predicate=delta_id,
                total="P or not P",
                exclusive="not (P and not P)",
                termination="finite boolean variable",
            )
            for index, delta_id in enumerate(ids)
        )
        return ProofCertificate(
            certificate_id="cert.boolean_partition",
            status=SOUND,
            engine="strict-proof-engine@0.1",
            obligations=obligations,
            artifacts=(
                {
                    "kind": "characteristic_identity",
                    "expression": "sum_b product_i delta_i^b_i * (1-delta_i)^(1-b_i) = 1",
                    "proof_method": "finite_boolean_partition_induction",
                    "delta_count": len(ids),
                },
            ),
            reason="finite product partition over boolean deltas",
        )

    def _unknown(self, certificate_id: str, reason: str) -> ProofCertificate:
        return ProofCertificate(certificate_id, UNKNOWN, "strict-proof-engine@0.1", (), (), reason)

    def _rejected(self, certificate_id: str, reason: str) -> ProofCertificate:
        return ProofCertificate(certificate_id, REJECTED_CERTIFICATE, "strict-proof-engine@0.1", (), (), reason)
