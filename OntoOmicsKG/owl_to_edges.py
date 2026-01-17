#!/usr/bin/env python3
import argparse
import re
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, OWL

# -----------------------------
# Config
# -----------------------------

# Weight annotations stored on owl:Axiom (edge reification)
WEIGHT_PROPS_DEFAULT = [
    "http://purl.obolibrary.org/obo/go.owl#hasScoreInteraction",
    "http://purl.obolibrary.org/obo/go.owl#hasExpressionValue",
]

# Keep only "signal" relations that are ASSERTED AS EDGES in your OWL file
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

# Fix malformed stringification that sometimes appears when parsing RDF/XML tags as strings:
#   http://.../go.owl#{http://.../go.owl#}hasGeneExpressionOA  ->  http://.../go.owl#hasGeneExpressionOA
_MALFORMED_IRI_RE = re.compile(r"^(.*)#\{.*\}(.+)$")


def normalize_iri(s: str) -> str:
    """Normalize occasional malformed IRI stringification."""
    m = _MALFORMED_IRI_RE.match(s)
    if m:
        return f"{m.group(1)}#{m.group(2)}"
    return s


def to_float(lit: Literal) -> float:
    try:
        return float(lit.toPython())
    except Exception:
        return float(str(lit))


def parse_weights_from_axioms(g: Graph, weight_props):
    """
    Extract weights stored as OWL2 axiom annotations.

    Returns:
        dict mapping (s,p,o) -> float(weight), with s/p/o as normalized strings.
    """
    weight_props = {URIRef(x) for x in weight_props}
    wmap = {}

    for ax in g.subjects(RDF.type, OWL.Axiom):
        s = g.value(ax, OWL.annotatedSource)
        p = g.value(ax, OWL.annotatedProperty)
        o = g.value(ax, OWL.annotatedTarget)
        if s is None or p is None or o is None:
            continue

        w_lit = None
        for wp in weight_props:
            w_lit = g.value(ax, wp)
            if w_lit is not None:
                break

        if not isinstance(w_lit, Literal):
            continue

        try:
            w = to_float(w_lit)
        except Exception:
            continue

        ss = normalize_iri(str(s))
        pp = normalize_iri(str(p))
        oo = normalize_iri(str(o))
        wmap[(ss, pp, oo)] = w

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

    # 1) Extract weights for edges via owl:Axiom
    wmap = parse_weights_from_axioms(g, weight_props)

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

    # 3) Write edges (skip OWL noise by relation whitelist)
    edges_written = 0
    weighted_edges_written = 0

    with open(out_tsv, "w", encoding="utf-8") as f:
        for s, p, o in g:
            # Skip literal objects (we only want entity->entity edges)
            if isinstance(o, Literal):
                continue

            ss = normalize_iri(str(s))
            pp = normalize_iri(str(p))
            oo = normalize_iri(str(o))

            # Keep only meaningful ontology relations
            if pp not in keep_relations:
                continue

            w = wmap.get((ss, pp, oo), float(default_weight))
            if w != 0.0:
                weighted_edges_written += 1

            f.write(f"{ent_id(ss)}\t{rel_id(pp)}\t{ent_id(oo)}\t{w}\n")
            edges_written += 1

    # 4) Save mappings
    with open(out_tsv + ".entities.tsv", "w", encoding="utf-8") as f:
        for iri, idx in sorted(ent2id.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{iri}\n")

    with open(out_tsv + ".relations.tsv", "w", encoding="utf-8") as f:
        for iri, idx in sorted(rel2id.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{iri}\n")

    # 5) Stats
    axiom_weights_for_kept_rel = sum(1 for (_, p, _), _w in wmap.items() if p in keep_relations)

    print(f"Edges written: {edges_written}")
    print(f"Entities: {len(ent2id)}")
    print(f"Relations: {len(rel2id)}")
    print(f"Weighted edges written (w != 0): {weighted_edges_written}")
    print(f"Weighted axioms found (all): {len(wmap)}")
    print(f"Weighted axioms whose predicate is in KEEP_RELATIONS: {axiom_weights_for_kept_rel}")


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
# Edges written: 255804
# Entities: 9187
# Relations: 17
# Weighted edges written (w != 0): 120796
# Weighted axioms found (all): 126293
# Weighted axioms whose predicate is in KEEP_RELATIONS: 126293