import unittest

from logical_delta_matter.formal_domains import MatterTraits, classify_matter
from logical_delta_matter.proof_engine import REJECTED_CERTIFICATE, SOUND, StrictProofEngine


class ProofEngineTests(unittest.TestCase):
    def test_certifies_finite_boolean_trace(self):
        decision = classify_matter(
            MatterTraits(
                continuous=True,
                grouped=False,
                planetary=False,
                preserves_shape=False,
                preserves_volume=False,
                ionized=False,
            )
        )
        certificate = StrictProofEngine().certificate_for_decision(decision)
        self.assertEqual(certificate.status, SOUND)
        self.assertEqual(certificate.obligations[0].predicate, "Continuous(x)")

    def test_rejects_admitted_proof_artifact(self):
        decision = classify_matter(
            MatterTraits(
                continuous=True,
                grouped=False,
                planetary=False,
                preserves_shape=False,
                preserves_volume=False,
                ionized=False,
            )
        )
        certificate = StrictProofEngine().certificate_for_decision(decision, artifacts=[{"text": "Theorem x. Admitted."}])
        self.assertEqual(certificate.status, REJECTED_CERTIFICATE)

    def test_partition_certificate_contains_characteristic_identity(self):
        certificate = StrictProofEngine().partition_certificate(["a", "b"])
        self.assertEqual(certificate.status, SOUND)
        self.assertEqual(certificate.artifacts[0]["kind"], "characteristic_identity")


if __name__ == "__main__":
    unittest.main()
