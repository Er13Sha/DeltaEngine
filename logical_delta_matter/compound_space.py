from __future__ import annotations

from itertools import combinations, product
import json
from pathlib import Path
from typing import Any

from .chemistry import capacity_for_requirement, hill_formula, parse_formula, particle_requirement, shortage_for_requirement
from .storage import write_json

REPULSION_CLASS_BY_GROUP = {
    "Nonmetal": "nonmetal",
    "Halogen": "halogen",
    "Noble gas": "noble_gas",
    "Alkali metal": "alkali_metal",
    "Alkaline earth metal": "alkaline_earth_metal",
    "Metalloid": "metalloid",
    "Post-transition metal": "post_transition_metal",
    "Transition metal": "transition_metal",
    "Lanthanide": "lanthanide",
    "Actinide": "actinide",
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compute_compound_space(
    concept_path: str | Path,
    config_path: str | Path,
    out_path: str | Path,
) -> dict[str, Any]:
    concepts = load_json(concept_path)
    config = load_json(config_path)
    elements = concepts["elements"]
    elements_by_symbol = {element["symbol"]: element for element in elements}
    confirmed_by_formula = collect_confirmed(concepts.get("compounds", []))

    candidates: dict[str, dict[str, Any]] = {}
    if config["generation"].get("include_confirmed", True):
        for compound in concepts.get("compounds", []):
            formula = compound["formula"]
            candidates[f"confirmed:{compound['cid']}"] = {
                "candidate_id": f"confirmed:{compound['cid']}",
                "formula": formula,
                "atoms": safe_parse_formula(formula),
                "origin": "confirmed_pubchem",
                "confirmed": True,
                "confirmed_refs": [compound_ref(compound)],
            }

    for atoms in generate_atom_maps(config["generation"]):
        formula = hill_formula(atoms)
        key = f"generated:{formula}"
        candidates.setdefault(
            key,
            {
                "candidate_id": key,
                "formula": formula,
                "atoms": atoms,
                "origin": "katyushchik_generated",
                "confirmed": formula in confirmed_by_formula,
                "confirmed_refs": confirmed_by_formula.get(formula, []),
            },
        )

    rows = []
    for candidate in candidates.values():
        atoms = candidate["atoms"]
        if not atoms:
            continue
        rows.append(evaluate_candidate(candidate, elements_by_symbol, config))

    rows.sort(key=lambda row: (row["status"] != "possible", row["formula"], row["candidate_id"]))
    summary = summarize(rows)
    output = {
        "schema_version": "computed-compound-space@0.1",
        "rule_pack_id": config.get("rule_pack_id", "katyushchik_hypothesis"),
        "status": "research_hypothesis",
        "description": "Computed possible/impossible compound space under explicit finite rules. Not a claim about real chemistry.",
        "config": config,
        "decision_manifest": build_decision_manifest(config),
        "summary": summary,
        "candidates": rows,
    }
    write_json(out_path, output)
    return output


def safe_parse_formula(formula: str) -> dict[str, int]:
    try:
        return parse_formula(formula)
    except ValueError:
        return {}


def collect_confirmed(compounds: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for compound in compounds:
        out.setdefault(compound["formula"], []).append(compound_ref(compound))
    return out


def compound_ref(compound: dict[str, Any]) -> dict[str, Any]:
    return {
        "cid": compound["cid"],
        "query": compound["query"],
        "iupac_name": compound.get("iupac_name", ""),
        "source_id": compound["verified"]["source_id"],
    }


def generate_atom_maps(generation: dict[str, Any]) -> list[dict[str, int]]:
    symbols = [str(symbol) for symbol in generation["elements"]]
    max_total_atoms = int(generation.get("max_total_atoms", 6))
    max_distinct = int(generation.get("max_distinct_elements", 3))
    max_count = int(generation.get("max_count_per_element", 4))
    rows: list[dict[str, int]] = []

    for distinct in range(1, max_distinct + 1):
        for combo in combinations(symbols, distinct):
            for counts in product(range(1, max_count + 1), repeat=distinct):
                total = sum(counts)
                if total < 2 or total > max_total_atoms:
                    continue
                rows.append({symbol: count for symbol, count in zip(combo, counts)})
    return rows


def evaluate_candidate(candidate: dict[str, Any], elements_by_symbol: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    atoms = candidate["atoms"]
    reasons: list[str] = []
    unknown = [symbol for symbol in atoms if symbol not in elements_by_symbol]
    if unknown:
        reasons.append(f"unknown_elements:{','.join(sorted(unknown))}")
        requirement = {"protons": 0, "neutrons": 0, "electrons": 0}
    else:
        requirement = particle_requirement(atoms, elements_by_symbol)

    rules = config.get("rules", {})
    stock = config.get("particle_stock", {})
    total_atoms = sum(atoms.values())
    min_atoms = int(rules.get("min_atoms_for_compound", 2))
    is_compound = total_atoms >= min_atoms
    if not is_compound:
        reasons.append("not_a_compound_single_atom")

    repulsion_classes = sorted({repulsion_class(elements_by_symbol[symbol]) for symbol in atoms if symbol in elements_by_symbol})
    homonuclear_allowed = not (
        rules.get("same_repulsion_homonuclear_impossible", True) and len(atoms) == 1 and total_atoms > 1
    )
    if not homonuclear_allowed:
        reasons.append("same_repulsion_homonuclear_without_binding_mechanism")

    min_classes = int(rules.get("min_repulsion_classes_for_generated", 2))
    class_diversity_allowed = candidate["origin"] != "katyushchik_generated" or len(repulsion_classes) >= min_classes
    if candidate["origin"] == "katyushchik_generated":
        if not class_diversity_allowed:
            reasons.append("insufficient_repulsion_class_diversity")

    noble_gas_allowed = not (
        rules.get("noble_gas_requires_confirmation", True) and "noble_gas" in repulsion_classes and not candidate["confirmed"]
    )
    if not noble_gas_allowed:
        reasons.append("noble_gas_generated_without_confirmation")

    capacity = capacity_for_requirement(requirement, stock)
    resource_available = not rules.get("resource_limited", True) or capacity > 0
    if not resource_available:
        reasons.append("resource_insufficient")

    status = "possible" if not reasons else "impossible"
    if candidate["confirmed"] and status == "impossible":
        theory_relation = "confirmed_but_impossible_under_current_research_rules_or_resources"
    elif candidate["confirmed"]:
        theory_relation = "confirmed_and_possible_under_current_research_rules"
    else:
        theory_relation = "computed_only"

    return {
        **candidate,
        "status": status,
        "reasons": reasons,
        "theory_relation": theory_relation,
        "repulsion_classes": repulsion_classes,
        "decision_trace": build_decision_trace(
            {
                "known_elements": not unknown,
                "is_compound": is_compound,
                "homonuclear_allowed": homonuclear_allowed,
                "class_diversity_allowed": class_diversity_allowed,
                "noble_gas_allowed": noble_gas_allowed,
                "resource_available": resource_available,
            }
        ),
        "particle_requirement": requirement,
        "capacity": capacity,
        "shortage": shortage_for_requirement(requirement, stock),
        "source": {
            "source_id": "katyushchik_hypothesis.compound_space",
            "rule_pack_id": config.get("rule_pack_id", "katyushchik_hypothesis"),
            "verified_refs": candidate.get("confirmed_refs", []),
        },
    }


def repulsion_class(element: dict[str, Any]) -> str:
    group = element.get("group_block", "")
    return REPULSION_CLASS_BY_GROUP.get(group, group.lower().replace(" ", "_") or "unknown")


def build_decision_trace(values: dict[str, bool]) -> list[dict[str, Any]]:
    specs = [
        ("compound.known_elements", "KnownElements(formula)", "known_elements", "unknown_elements"),
        ("compound.min_atoms", "AtomCount(formula) >= min_atoms_for_compound", "is_compound", "not_a_compound"),
        (
            "compound.homonuclear_allowed",
            "not SameRepulsionHomonuclear(formula) or BindingMechanismConfirmed(formula)",
            "homonuclear_allowed",
            "same_repulsion_homonuclear",
        ),
        (
            "compound.repulsion_class_diversity",
            "Confirmed(formula) or RepulsionClassCount(formula) >= min_repulsion_classes_for_generated",
            "class_diversity_allowed",
            "insufficient_repulsion_class_diversity",
        ),
        (
            "compound.noble_gas_allowed",
            "not ContainsNobleGas(formula) or Confirmed(formula)",
            "noble_gas_allowed",
            "noble_gas_generated_without_confirmation",
        ),
        ("compound.resource_available", "ParticleStockCovers(formula)", "resource_available", "resource_insufficient"),
    ]
    return [
        {
            "delta_id": delta_id,
            "predicate": predicate,
            "value": bool(values[key]),
            "branches": {"true": "continue", "false": false_branch},
        }
        for delta_id, predicate, key, false_branch in specs
    ]


def build_decision_manifest(config: dict[str, Any]) -> list[dict[str, Any]]:
    domain = "CompoundCandidate"
    obligations = {
        "total": "The decider reads finite formula atoms, finite element records, finite integer stock and finite rule flags.",
        "exclusive": "Every predicate is evaluated as a boolean; true and false branches cannot both hold.",
        "termination": "Formula parsing, element lookup and particle arithmetic are bounded by finite formula length and finite candidate count.",
    }
    return [
        {
            "delta_id": "compound.known_elements",
            "domain": domain,
            "predicate": "KnownElements(formula)",
            "decider": "all symbols in parsed formula exist in confirmed element catalog",
            "obligations": obligations,
            "branches": {"true": "continue", "false": "unknown_elements"},
        },
        {
            "delta_id": "compound.min_atoms",
            "domain": domain,
            "predicate": f"AtomCount(formula) >= {int(config.get('rules', {}).get('min_atoms_for_compound', 2))}",
            "decider": "sum parsed atom counts and compare with configured integer threshold",
            "obligations": obligations,
            "branches": {"true": "continue", "false": "not_a_compound"},
        },
        {
            "delta_id": "compound.homonuclear_allowed",
            "domain": domain,
            "predicate": "not SameRepulsionHomonuclear(formula) or BindingMechanismConfirmed(formula)",
            "decider": "reject multi-atom one-element formulas when same-repulsion homonuclear ban is enabled",
            "obligations": obligations,
            "branches": {"true": "continue", "false": "same_repulsion_homonuclear"},
        },
        {
            "delta_id": "compound.repulsion_class_diversity",
            "domain": domain,
            "predicate": "Confirmed(formula) or RepulsionClassCount(formula) >= min_repulsion_classes_for_generated",
            "decider": "map element group blocks to finite repulsion classes and count unique classes",
            "obligations": obligations,
            "branches": {"true": "continue", "false": "insufficient_repulsion_class_diversity"},
        },
        {
            "delta_id": "compound.noble_gas_allowed",
            "domain": domain,
            "predicate": "not ContainsNobleGas(formula) or Confirmed(formula)",
            "decider": "check whether any element class is noble_gas and whether the candidate is confirmed",
            "obligations": obligations,
            "branches": {"true": "continue", "false": "noble_gas_generated_without_confirmation"},
        },
        {
            "delta_id": "compound.resource_available",
            "domain": domain,
            "predicate": "ParticleStockCovers(formula)",
            "decider": "calculate exact integer p/n/e requirement and compare with configured stock",
            "obligations": obligations,
            "branches": {"true": "possible", "false": "resource_insufficient"},
        },
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    possible = [row for row in rows if row["status"] == "possible"]
    impossible = [row for row in rows if row["status"] == "impossible"]
    reason_counts: dict[str, int] = {}
    for row in impossible:
        for reason in row["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "total": len(rows),
        "possible": len(possible),
        "impossible": len(impossible),
        "confirmed": sum(1 for row in rows if row["confirmed"]),
        "generated": sum(1 for row in rows if row["origin"] == "katyushchik_generated"),
        "impossible_reason_counts": dict(sorted(reason_counts.items())),
    }
