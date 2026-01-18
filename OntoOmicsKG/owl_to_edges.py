#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, OWL

WEIGHT_PROPS_DEFAULT = [
    "http://purl.obolibrary.org/obo/go.owl#hasScoreInteraction",
    "http://purl.obolibrary.org/obo/go.owl#hasExpressionValue",
]

KEEP_RELATIONS = {
    "http://purl.obolibrary.org/obo/go.owl#hasGeneExpressionOA",
    "http://purl.obolibrary.org/obo/go.owl#partOfPathway",
    "http://purl.obolibrary.org/obo/go.owl#containedIn",
    "http://purl.obolibrary.org/obo/go.owl#isAssociatedWithGO",
    "http://purl.obolibrary.org/obo/go.owl#participatedIn",
    "http://purl.obolibrary.org/obo/go.owl#isAssociatedWithProteinPathway",
    "http://purl.obolibrary.org/obo/go.owl#hasFunctionalInteractionWith",
    "http://purl.obolibrary.org/obo/go.owl#hasPhysicalInteractionWith",
    "http://purl.obolibrary.org/obo/go.owl#isAGO",
    "http://purl.obolibrary.org/obo/go.owl#hasGeneticInteractionWith",
    "http://purl.obolibrary.org/obo/go.owl#hasPartGO",
    "http://purl.obolibrary.org/obo/go.owl#hasModification",
    "http://purl.obolibrary.org/obo/go.owl#partOfGO",
    "http://purl.obolibrary.org/obo/go.owl#negativelyRegulatesGO",
    "http://purl.obolibrary.org/obo/go.owl#regulatesGO",
    "http://purl.obolibrary.org/obo/go.owl#positivelyRegulatesGO",
    "http://purl.obolibrary.org/obo/go.owl#occursInGO",
}

_MALFORMED_IRI_RE = re.compile(r"^(.*)#\{.*\}(.+)$")

def normalize_iri(s: str) -> str:
    m = _MALFORMED_IRI_RE.match(s)
    if m:
        return f"{m.group(1)}#{m.group(2)}"
    return s

def to_float(lit: Literal) -> float:
    # Handles xsd:decimal, xsd:float, xsd:double, and numeric-ish strings like "-1.9"
    return float(lit.toPython())

def parse_weights_from_axioms_all(g: Graph, weight_props):
    """
    Extract ALL numeric weights stored as OWL2 axiom annotations.

    Returns:
        dict mapping (s,p,o) -> list[float]
    """
    weight_props = {URIRef(x) for x in weight_props}
    wmap = defaultdict(list)

    for ax in g.subjects(RDF.type, OWL.Axiom):
        s = g.value(ax, OWL.annotatedSource)
        p = g.value(ax, OWL.annotatedProperty)
        o = g.value(ax, OWL.annotatedTarget)
        if s is None or p is None or o is None:
            continue

        ss = normalize_iri(str(s))
        pp = normalize_iri(str(p))
        oo = normalize_iri(str(o))

        for wp in weight_props:
            for lit in g.objects(ax, wp):
                if not isinstance(lit, Literal):
                    continue
                try:
                    w = to_float(lit)
                except Exception:
                    # skips "-", "NA", etc.
                    continue
                wmap[(ss, pp, oo)].append(w)

    return wmap

def convert_owl_to_weighted_edgelist(
    owl_file: str,
    out_tsv: str,
    default_weight: float = 0.0,
    weight_props=None,
    keep_relations=None,
):
    g = Graph()
    g.parse(owl_file, format="xml")

    if weight_props is None:
        weight_props = WEIGHT_PROPS_DEFAULT
    if keep_relations is None:
        keep_relations = KEEP_RELATIONS

    # 1) Extract ALL weights per edge via owl:Axiom
    wmap = parse_weights_from_axioms_all(g, weight_props)

    # 2) ID maps
    ent2id = {}
    rel2id = {}

    def ent_id(x: str) -> int:
        if x not in ent2id:
            ent2id[x] = len(ent2id)
        return ent2id[x]

    def rel_id(x: str) -> int:
        if x not in rel2id:
            rel2id[x] = len(rel2id)
        return rel2id[x]

    edges_written = 0
    weighted_edges_written = 0

    with open(out_tsv, "w", encoding="utf-8") as f:
        for s, p, o in g:
            if isinstance(o, Literal):
                continue

            ss = normalize_iri(str(s))
            pp = normalize_iri(str(p))
            oo = normalize_iri(str(o))

            if pp not in keep_relations:
                continue

            weights = wmap.get((ss, pp, oo), [])

            # IMPORTANT: one output row per weight
            if weights:
                for w in weights:
                    if w != 0.0:
                        weighted_edges_written += 1
                    f.write(f"{ent_id(ss)}\t{rel_id(pp)}\t{ent_id(oo)}\t{w}\n")
                    edges_written += 1
            else:
                # if no weight annotations exist, still emit the edge once
                f.write(f"{ent_id(ss)}\t{rel_id(pp)}\t{ent_id(oo)}\t{float(default_weight)}\n")
                edges_written += 1

    # Save mappings
    with open(out_tsv + ".entities.tsv", "w", encoding="utf-8") as f:
        for iri, idx in sorted(ent2id.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{iri}\n")

    with open(out_tsv + ".relations.tsv", "w", encoding="utf-8") as f:
        for iri, idx in sorted(rel2id.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{iri}\n")

    axiom_weights_for_kept_rel = sum(
        len(ws) for ((_, p, _), ws) in wmap.items() if p in keep_relations
    )

    print(f"Edges written: {edges_written}")
    print(f"Entities: {len(ent2id)}")
    print(f"Relations: {len(rel2id)}")
    print(f"Weighted edges written (w != 0): {weighted_edges_written}")
    print(f"Weighted axiom values found (all numeric values): {sum(len(v) for v in wmap.values())}")
    print(f"Weighted axiom values whose predicate is in KEEP_RELATIONS: {axiom_weights_for_kept_rel}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input OWL file (RDF/XML)")
    ap.add_argument("--out", dest="out", required=True, help="Output TSV (head rel tail weight)")
    ap.add_argument("--default-weight", type=float, default=0.0, help="Default weight for unweighted edges")
    ap.add_argument(
        "--weight-prop",
        action="append",
        default=None,
        help="Repeatable. Defaults to hasScoreInteraction + hasExpressionValue",
    )
    args = ap.parse_args()

    convert_owl_to_weighted_edgelist(
        owl_file=args.inp,
        out_tsv=args.out,
        default_weight=args.default_weight,
        weight_props=args.weight_prop if args.weight_prop else WEIGHT_PROPS_DEFAULT,
        keep_relations=KEEP_RELATIONS,
    )

if __name__ == "__main__":
    main()


# python owl_to_edges.py --in GSE54514_enriched_ontology_degfilter_v2.11_avgExpression_ovp0.2_ng4.owl --out edges.filtered.tsv --default-weight 0
# Edges written: 265330
# Entities: 9187
# Relations: 17
# Weighted edges written (w != 0): 131350
# Weighted axioms found (all): 136847
# Weighted axioms whose predicate is in KEEP_RELATIONS: 136847