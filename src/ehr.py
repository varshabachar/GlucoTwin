"""
Step 3 - The STATIC stream: a synthetic Electronic Health Record for every participant.

Why synthetic? The challenge sandbox rules (DPDP Act / HIPAA) forbid real patient records.
What is real: each participant's sex and HbA1c (from Demographics.csv).
What is generated: age, body measurements, lab panel, past diagnoses, family history,
medications and genetic markers - all made consistent with that person's real HbA1c.

Records are written as FHIR R4 Bundles (the interoperability standard used by India's ABDM),
then parsed back into model features - exactly how a hospital system would feed the twin.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config

EHR_FEATURES = [
    "age", "sex_male", "bmi", "hba1c", "fasting_glucose", "triglycerides", "hdl", "ldl",
    "sbp", "dbp", "dx_prediabetes", "dx_hypertension", "dx_dyslipidemia",
    "family_hx_t2d", "on_metformin", "on_statin", "prs",
]

REFERENCE_YEAR = 2020  # the BIG IDEAs recordings are dated 2020

# Type-2-diabetes risk variants. Allele frequencies and odds ratios are ILLUSTRATIVE values of the
# right order of magnitude from published GWAS - replace with population-specific values if you have them.
SNPS = [
    # gene, rsID, risk allele, risk-allele frequency, odds ratio per allele
    ("TCF7L2", "rs7903146", "T", 0.30, 1.40),
    ("FTO", "rs9939609", "A", 0.40, 1.15),
    ("PPARG", "rs1801282", "C", 0.88, 1.14),
    ("KCNJ11", "rs5219", "T", 0.35, 1.15),
    ("SLC30A8", "rs13266634", "C", 0.70, 1.12),
]

LOINC = {
    "hba1c": ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "%"),
    "fasting_glucose": ("1558-6", "Fasting glucose [Mass/volume] in Serum or Plasma", "mg/dL"),
    "bmi": ("39156-5", "Body mass index (BMI) [Ratio]", "kg/m2"),
    "weight": ("29463-7", "Body weight", "kg"),
    "height": ("8302-2", "Body height", "cm"),
    "sbp": ("8480-6", "Systolic blood pressure", "mm[Hg]"),
    "dbp": ("8462-4", "Diastolic blood pressure", "mm[Hg]"),
    "triglycerides": ("2571-8", "Triglyceride [Mass/volume] in Serum or Plasma", "mg/dL"),
    "hdl": ("2085-9", "Cholesterol in HDL [Mass/volume] in Serum or Plasma", "mg/dL"),
    "ldl": ("13457-7", "Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation", "mg/dL"),
}
SNOMED = {
    "dx_prediabetes": ("714628002", "Prediabetes"),
    "dx_hypertension": ("59621000", "Essential hypertension"),
    "dx_dyslipidemia": ("55822004", "Hyperlipidemia"),
}
RXNORM = {"on_metformin": ("6809", "metformin"), "on_statin": ("83367", "atorvastatin")}


# ------------------------------------------------------------------ generate
def _generate_values(pid: str, sex: str, hba1c: float) -> dict:
    rng = np.random.default_rng(int(pid) * 7919 + config.RANDOM_SEED)
    male = sex.upper().startswith("M")
    age = int(rng.integers(35, 66))
    height = rng.normal(175, 7) if male else rng.normal(162, 6)
    bmi = float(np.clip(rng.normal(26.5 + 2.5 * (hba1c - 5.6), 3.5), 19, 40))
    weight = bmi * (height / 100) ** 2
    fpg = float(np.clip(92 + 18 * (hba1c - 5.6) + rng.normal(0, 6), 75, 125))
    tg = float(np.exp(rng.normal(np.log(130 + 7.5 * (bmi - 26)), 0.35)))
    hdl = float(np.clip((rng.normal(45, 10) if male else rng.normal(55, 12)) - 0.8 * (bmi - 26), 25, 95))
    ldl = float(np.clip(rng.normal(120, 30), 50, 220))
    sbp = float(112 + 0.45 * (age - 40) + 0.9 * (bmi - 25) + rng.normal(0, 10))
    dbp = float(72 + 0.5 * (bmi - 25) + rng.normal(0, 7))
    v = dict(
        age=age, sex_male=int(male), height=round(height, 1), weight=round(weight, 1), bmi=round(bmi, 1),
        hba1c=float(hba1c), fasting_glucose=round(fpg), triglycerides=round(tg), hdl=round(hdl),
        ldl=round(ldl), sbp=round(sbp), dbp=round(dbp),
    )
    v["dx_prediabetes"] = int(hba1c >= 5.7 or fpg >= 100)                     # ADA criteria
    v["dx_hypertension"] = int(sbp >= 130 or dbp >= 80)                        # ACC/AHA 2017 stage 1+
    v["dx_dyslipidemia"] = int(ldl >= 160 or tg >= 150 or hdl < (40 if male else 50))
    v["family_hx_t2d"] = int(rng.random() < 0.25 + 0.15 * (hba1c >= 5.7))
    v["on_metformin"] = int(v["dx_prediabetes"] and rng.random() < 0.15)
    v["on_statin"] = int(v["dx_dyslipidemia"] and rng.random() < 0.5)
    geno, prs_raw = {}, 0.0
    mean = sum(2 * p * np.log(o) for *_, p, o in SNPS)
    var = sum(2 * p * (1 - p) * np.log(o) ** 2 for *_, p, o in SNPS)
    for gene, rs, allele, p, odds in SNPS:
        dose = int(rng.binomial(2, p))
        geno[rs] = dict(gene=gene, risk_allele=allele, risk_allele_copies=dose)
        prs_raw += dose * np.log(odds)
    v["prs"] = round(float((prs_raw - mean) / np.sqrt(var)), 2)  # standardised polygenic risk score
    v["genotypes"] = geno
    v["dx_onset_years_ago"] = {k: int(rng.integers(1, 8)) for k in SNOMED if v[k]}
    return v


def _obs(pid, key, value, idx):
    code, display, unit = LOINC[key]
    return {
        "resource": {
            "resourceType": "Observation", "id": f"{pid}-obs-{idx}", "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                      "code": "laboratory" if key not in ("bmi", "weight", "height", "sbp", "dbp") else "vital-signs"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}], "text": key},
            "subject": {"reference": f"Patient/{pid}"},
            "valueQuantity": {"value": value, "unit": unit, "system": "http://unitsofmeasure.org", "code": unit},
        }
    }


def make_fhir_bundle(pid: str, v: dict) -> dict:
    entries = [{
        "resource": {
            "resourceType": "Patient", "id": pid,
            "gender": "male" if v["sex_male"] else "female",
            "birthDate": f"{REFERENCE_YEAR - v['age']}-01-01",
            "meta": {"tag": [{"code": "SYNTHETIC", "display": "Synthetic record - no real patient data"}]},
        }
    }]
    for i, key in enumerate(LOINC):
        entries.append(_obs(pid, key, v[key], i))
    for key, (code, display) in SNOMED.items():
        if v[key]:
            entries.append({"resource": {
                "resourceType": "Condition", "id": f"{pid}-{key}",
                "clinicalStatus": {"coding": [{"code": "active"}]},
                "code": {"coding": [{"system": "http://snomed.info/sct", "code": code, "display": display}], "text": key},
                "subject": {"reference": f"Patient/{pid}"},
                "onsetDateTime": f"{REFERENCE_YEAR - v['dx_onset_years_ago'][key]}-06-01",
            }})
    if v["family_hx_t2d"]:
        entries.append({"resource": {
            "resourceType": "FamilyMemberHistory", "id": f"{pid}-fh", "status": "completed",
            "patient": {"reference": f"Patient/{pid}"},
            "relationship": {"coding": [{"code": "PRN", "display": "parent"}]},
            "condition": [{"code": {"coding": [{"system": "http://snomed.info/sct", "code": "44054006",
                                                "display": "Type 2 diabetes mellitus"}]}}],
        }})
    for key, (code, name) in RXNORM.items():
        if v[key]:
            entries.append({"resource": {
                "resourceType": "MedicationStatement", "id": f"{pid}-{key}", "status": "active",
                "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                                                          "code": code, "display": name}], "text": key},
                "subject": {"reference": f"Patient/{pid}"},
            }})
    for rs, g in v["genotypes"].items():
        entries.append({"resource": {
            "resourceType": "Observation", "id": f"{pid}-{rs}", "status": "final",
            "category": [{"coding": [{"code": "laboratory"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": "69548-6", "display": "Genetic variant assessment"}],
                     "text": "genetic_variant"},
            "subject": {"reference": f"Patient/{pid}"},
            "component": [
                {"code": {"text": "gene"}, "valueString": g["gene"]},
                {"code": {"text": "rsid"}, "valueString": rs},
                {"code": {"text": "risk_allele"}, "valueString": g["risk_allele"]},
                {"code": {"text": "risk_allele_copies"}, "valueInteger": g["risk_allele_copies"]},
            ],
        }})
    entries.append({"resource": {
        "resourceType": "Observation", "id": f"{pid}-prs", "status": "final",
        "code": {"text": "prs", "coding": [{"display": "Type 2 diabetes polygenic risk score (z)"}]},
        "subject": {"reference": f"Patient/{pid}"}, "valueQuantity": {"value": v["prs"], "unit": "z"},
    }})
    return {"resourceType": "Bundle", "id": f"ehr-{pid}", "type": "collection", "entry": entries}


# ------------------------------------------------------------------ ingest (FHIR -> features)
def bundle_to_features(bundle: dict) -> dict:
    """Parse a FHIR Bundle into the flat feature vector the model uses."""
    f = {k: 0 for k in EHR_FEATURES}
    genes = []
    for e in bundle["entry"]:
        r = e["resource"]
        rt = r["resourceType"]
        if rt == "Patient":
            f["sex_male"] = int(r["gender"] == "male")
            f["age"] = REFERENCE_YEAR - int(r["birthDate"][:4])
        elif rt == "Observation":
            key = r["code"].get("text")
            if key in EHR_FEATURES and "valueQuantity" in r:
                f[key] = r["valueQuantity"]["value"]
            elif key == "genetic_variant":
                comp = {c["code"]["text"]: next(v for k, v in c.items() if k.startswith("value")) for c in r["component"]}
                genes.append(f"{comp['gene']} {comp['rsid']}: {comp['risk_allele_copies']} risk allele(s)")
        elif rt == "Condition":
            f[r["code"]["text"]] = 1
        elif rt == "MedicationStatement":
            f[r["medicationCodeableConcept"]["text"]] = 1
        elif rt == "FamilyMemberHistory":
            f["family_hx_t2d"] = 1
    f["_genotypes"] = genes
    return f


def build_ehr(demo: pd.DataFrame, out_dir: Path | None = None) -> pd.DataFrame:
    """Generate, save and re-ingest a FHIR record for every participant -> one row per patient."""
    out_dir = out_dir or (config.ARTIFACTS_DIR / "ehr")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid, r in demo.iterrows():
        v = _generate_values(pid, r["gender"], float(r["hba1c"]))
        bundle = make_fhir_bundle(pid, v)
        (out_dir / f"patient_{pid}.json").write_text(json.dumps(bundle, indent=2))
        f = bundle_to_features(json.loads((out_dir / f"patient_{pid}.json").read_text()))
        f["pid"] = pid
        f["genotypes"] = "; ".join(f.pop("_genotypes"))
        rows.append(f)
    ehr = pd.DataFrame(rows).set_index("pid")
    ehr.to_csv(out_dir / "ehr_summary.csv")
    return ehr
