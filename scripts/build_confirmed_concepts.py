from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import argparse
from pathlib import Path
from typing import Any

PUBCHEM_PERIODIC_TABLE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/JSON"
PUBCHEM_COMPOUND_PROPERTY = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/"
    "property/MolecularFormula,IUPACName,CanonicalSMILES,InChIKey/JSON"
)
DEFAULT_COMPOUND_SEED = Path("data/compound-seed.json")

ELEMENT_RU_ALIASES = {
    "H": ["водород", "водорода", "гидроген"],
    "He": ["гелий", "гелия"],
    "Li": ["литий", "лития"],
    "Be": ["бериллий", "бериллия"],
    "B": ["бор", "бора"],
    "C": ["углерод", "углерода"],
    "N": ["азот", "азота"],
    "O": ["кислород", "кислорода"],
    "F": ["фтор", "фтора"],
    "Ne": ["неон", "неона"],
    "Na": ["натрий", "натрия"],
    "Mg": ["магний", "магния"],
    "Al": ["алюминий", "алюминия"],
    "Si": ["кремний", "кремния"],
    "P": ["фосфор", "фосфора"],
    "S": ["сера", "серы", "серу"],
    "Cl": ["хлор", "хлора"],
    "Ar": ["аргон", "аргона"],
    "K": ["калий", "калия"],
    "Ca": ["кальций", "кальция"],
    "Sc": ["скандий", "скандия"],
    "Ti": ["титан", "титана"],
    "V": ["ванадий", "ванадия"],
    "Cr": ["хром", "хрома"],
    "Mn": ["марганец", "марганца"],
    "Fe": ["железо", "железа"],
    "Co": ["кобальт", "кобальта"],
    "Ni": ["никель", "никеля"],
    "Cu": ["медь", "меди"],
    "Zn": ["цинк", "цинка"],
    "Ag": ["серебро", "серебра"],
    "Au": ["золото", "золота"],
    "Pb": ["свинец", "свинца"],
    "Ga": ["галлий"],
    "Ge": ["германий"],
    "As": ["мышьяк"],
    "Se": ["селен"],
    "Br": ["бром"],
    "Kr": ["криптон"],
    "Rb": ["рубидий"],
    "Sr": ["стронций"],
    "Y": ["иттрий"],
    "Zr": ["цирконий"],
    "Nb": ["ниобий"],
    "Mo": ["молибден"],
    "Tc": ["технеций"],
    "Ru": ["рутений"],
    "Rh": ["родий"],
    "Pd": ["палладий"],
    "Cd": ["кадмий"],
    "In": ["индий"],
    "Sn": ["олово"],
    "Sb": ["сурьма"],
    "Te": ["теллур"],
    "I": ["йод", "иод"],
    "Xe": ["ксенон"],
    "Cs": ["цезий"],
    "Ba": ["барий"],
    "La": ["лантан"],
    "Ce": ["церий"],
    "Pr": ["празеодим"],
    "Nd": ["неодим"],
    "Pm": ["прометий"],
    "Sm": ["самарий"],
    "Eu": ["европий"],
    "Gd": ["гадолиний"],
    "Tb": ["тербий"],
    "Dy": ["диспрозий"],
    "Ho": ["гольмий"],
    "Er": ["эрбий"],
    "Tm": ["тулий"],
    "Yb": ["иттербий"],
    "Lu": ["лютеций"],
    "Hf": ["гафний"],
    "Ta": ["тантал"],
    "W": ["вольфрам"],
    "Re": ["рений"],
    "Os": ["осмий"],
    "Ir": ["иридий"],
    "Pt": ["платина"],
    "Hg": ["ртуть"],
    "Tl": ["таллий"],
    "Bi": ["висмут"],
    "Po": ["полоний"],
    "At": ["астат"],
    "Rn": ["радон"],
    "Fr": ["франций"],
    "Ra": ["радий"],
    "Ac": ["актиний"],
    "Th": ["торий"],
    "Pa": ["протактиний"],
    "U": ["уран"],
    "Np": ["нептуний"],
    "Pu": ["плутоний"],
    "Am": ["америций"],
    "Cm": ["кюрий"],
    "Bk": ["берклий"],
    "Cf": ["калифорний"],
    "Es": ["эйнштейний"],
    "Fm": ["фермий"],
    "Md": ["менделевий"],
    "No": ["нобелий"],
    "Lr": ["лоуренсий"],
    "Rf": ["резерфордий"],
    "Db": ["дубний"],
    "Sg": ["сиборгий"],
    "Bh": ["борий"],
    "Hs": ["хассий"],
    "Mt": ["мейтнерий"],
    "Ds": ["дармштадтий"],
    "Rg": ["рентгений"],
    "Cn": ["коперниций"],
    "Nh": ["нихоний"],
    "Fl": ["флеровий"],
    "Mc": ["московий"],
    "Lv": ["ливерморий"],
    "Ts": ["теннессин"],
    "Og": ["оганесон"],
}

COMPOUND_TARGETS = [
    {"query": "water", "aliases_ru": ["вода", "воды", "воду", "жидкая вода"], "phase": "liquid", "family": "bio"},
    {"query": "hydrogen", "aliases_ru": ["молекулярный водород"], "phase": "gas", "family": "simple"},
    {"query": "oxygen", "aliases_ru": ["молекулярный кислород"], "phase": "gas", "family": "simple"},
    {"query": "nitrogen", "aliases_ru": ["молекулярный азот"], "phase": "gas", "family": "simple"},
    {"query": "carbon dioxide", "aliases_ru": ["диоксид углерода", "углекислый газ", "углекислота"], "phase": "gas", "family": "bio"},
    {"query": "methane", "aliases_ru": ["метан", "метана"], "phase": "gas", "family": "bio"},
    {"query": "ammonia", "aliases_ru": ["аммиак", "аммиака"], "phase": "gas", "family": "bio"},
    {"query": "sodium chloride", "aliases_ru": ["хлорид натрия", "соль", "поваренная соль"], "phase": "solid", "family": "tech"},
    {"query": "silicon dioxide", "aliases_ru": ["диоксид кремния", "кремнезем", "силикатная база"], "phase": "solid", "family": "tech"},
    {"query": "glucose", "aliases_ru": ["глюкоза", "глюкозу", "сахар"], "phase": "solid", "family": "bio"},
    {"query": "glycine", "aliases_ru": ["глицин", "глицина", "аминокислота"], "phase": "solid", "family": "bio"},
    {"query": "ethanol", "aliases_ru": ["этанол", "спирт", "этиловый спирт"], "phase": "liquid", "family": "bio"},
    {"query": "acetic acid", "aliases_ru": ["уксусная кислота", "ацетатная кислота"], "phase": "liquid", "family": "bio"},
    {"query": "sodium hydroxide", "aliases_ru": ["гидроксид натрия", "едкий натр"], "phase": "solid", "family": "tech"},
    {"query": "hydrochloric acid", "aliases_ru": ["соляная кислота", "хлороводородная кислота"], "phase": "liquid", "family": "tech"},
    {"query": "sulfuric acid", "aliases_ru": ["серная кислота"], "phase": "liquid", "family": "tech"},
    {"query": "calcium carbonate", "aliases_ru": ["карбонат кальция", "мел", "известняк"], "phase": "solid", "family": "tech"},
    {"query": "magnesium oxide", "aliases_ru": ["оксид магния"], "phase": "solid", "family": "tech"},
    {"query": "ferric oxide", "aliases_ru": ["оксид железа", "оксид железа iii", "ржавчина"], "phase": "solid", "family": "tech"},
    {"query": "copper(II) sulfate", "aliases_ru": ["сульфат меди", "медный купорос"], "phase": "solid", "family": "tech"},
    {"query": "potassium chloride", "aliases_ru": ["хлорид калия"], "phase": "solid", "family": "tech"},
    {"query": "calcium phosphate", "aliases_ru": ["фосфат кальция"], "phase": "solid", "family": "bio"},
    {"query": "adenosine triphosphate", "aliases_ru": ["атф", "аденозинтрифосфат"], "phase": "solid", "family": "bio"},
]


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "LogicalDeltaCorrect/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def parse_atomic_mass(value: str) -> float:
    # PubChem values can contain brackets or ranges for synthetic elements.
    match = re.search(r"\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match else 0.0


def model_phase(standard_state: str) -> str:
    state = standard_state.lower()
    if state == "gas":
        return "gas"
    if state == "liquid":
        return "liquid"
    return "solid"


def build_elements() -> list[dict[str, Any]]:
    data = fetch_json(PUBCHEM_PERIODIC_TABLE)
    columns = data["Table"]["Columns"]["Column"]
    rows = []
    for raw in data["Table"]["Row"]:
        row = dict(zip(columns, raw["Cell"]))
        atomic_number = int(row["AtomicNumber"])
        atomic_mass = parse_atomic_mass(row["AtomicMass"])
        mass_number = int(round(atomic_mass))
        rows.append(
            {
                "symbol": row["Symbol"],
                "name_en": row["Name"],
                "atomic_number": atomic_number,
                "atomic_mass": row["AtomicMass"],
                "standard_state": row["StandardState"],
                "group_block": row["GroupBlock"],
                "aliases": [row["Symbol"], row["Name"], *ELEMENT_RU_ALIASES.get(row["Symbol"], [])],
                "verified": {
                    "source_id": "pubchem.periodic_table",
                    "fields": ["atomic_number", "symbol", "name_en", "atomic_mass", "standard_state", "group_block"],
                },
                "model": {
                    "mass_number": mass_number,
                    "neutrons": max(0, mass_number - atomic_number),
                    "phase": model_phase(row["StandardState"]),
                    "status": "engineering_assumption_from_rounded_atomic_mass",
                },
            }
        )
    return rows


def load_compound_targets(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return COMPOUND_TARGETS
    data = json.loads(path.read_text(encoding="utf-8"))
    targets = data.get("targets", [])
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"{path} must contain non-empty targets list.")
    return targets


def unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def build_compounds(targets: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    failures = []
    for target in targets:
        encoded = urllib.parse.quote(target["query"])
        url = PUBCHEM_COMPOUND_PROPERTY.format(name=encoded)
        try:
            data = fetch_json(url)
        except Exception as exc:
            failures.append(
                {
                    "query": target.get("query"),
                    "error": str(exc),
                    "source_url": url,
                }
            )
            continue
        props = data["PropertyTable"]["Properties"][0]
        formula = target.get("formula") or props["MolecularFormula"]
        rows.append(
            {
                "cid": props["CID"],
                "query": target["query"],
                "formula": formula,
                "pubchem_molecular_formula": props["MolecularFormula"],
                "iupac_name": props.get("IUPACName", ""),
                "canonical_smiles": props.get("CanonicalSMILES", ""),
                "inchi_key": props.get("InChIKey", ""),
                "aliases": unique([target["query"], formula, props["MolecularFormula"], *target.get("aliases_ru", [])]),
                "verified": {
                    "source_id": "pubchem.compound_property",
                    "fields": ["cid", "pubchem_molecular_formula", "iupac_name", "canonical_smiles", "inchi_key"],
                },
                "model": {
                    "phase": target.get("phase", "solid"),
                    "family": target.get("family", "unclassified"),
                    "status": "local_prototype_phase_family",
                },
            }
        )
        time.sleep(0.12)
    return rows, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Build confirmed concept snapshot from PubChem.")
    parser.add_argument("--compound-seed", default=str(DEFAULT_COMPOUND_SEED), help="JSON file with compound targets.")
    parser.add_argument("--out", default="data/confirmed-concepts.json", help="Output JSON path.")
    args = parser.parse_args()

    targets = load_compound_targets(Path(args.compound_seed))
    compounds, failures = build_compounds(targets)
    out = {
        "schema_version": "confirmed-concepts@0.1",
        "metadata": {
            "generated_by": "scripts/build_confirmed_concepts.py",
            "sources": {
                "pubchem.periodic_table": PUBCHEM_PERIODIC_TABLE,
                "pubchem.compound_property": "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
            },
            "notes": [
                "Verified fields are copied from PubChem PUG-REST.",
                "Russian aliases and phase/family fields are local prototype metadata.",
                "model.neutrons is an engineering assumption from rounded atomic mass, not a verified isotope claim.",
                "The element table is complete for the PubChem periodic table snapshot.",
                "Compounds are imported from data/compound-seed.json; PubChem as a whole is too large to vendor in the repo.",
            ],
            "compound_seed": str(args.compound_seed),
            "compound_target_count": len(targets),
            "compound_failure_count": len(failures),
            "compound_failures": failures,
        },
        "elements": build_elements(),
        "compounds": compounds,
    }
    path = Path(args.out)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path} with {len(out['elements'])} elements and {len(out['compounds'])} compounds")
    if failures:
        print(f"skipped {len(failures)} compounds; see metadata.compound_failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
