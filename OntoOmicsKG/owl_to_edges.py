import argparse
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, OWL

WEIGHT_PROPS_DEFAULT = [
    "http://purl.obolibrary.org/obo/go.owl#hasScoreInteraction",
    "http://purl.obolibrary.org/obo/go.owl#hasExpressionValue",
]

def to_float(lit: Literal):
    try:
        v = lit.toPython()
        return float(v)
    except Exception:
        return float(str(lit))

def parse_weights_from_axioms(g: Graph, weight_props):
    """
    Returns dict: (s,p,o) -> weight
    where (s,p,o) are string IRIs / blank IDs.
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
        if w_lit is None:
            continue

        try:
            w = to_float(w_lit)
        except Exception:
            continue

        wmap[(str(s), str(p), str(o))] = w

    return wmap

def convert_owl_to_weighted_edgelist(owl_file, out_tsv, default_weight=0.0, weight_props=None):
    g = Graph()
    g.parse(owl_file, format="xml")

    if weight_props is None:
        weight_props = WEIGHT_PROPS_DEFAULT

    # 1) get weights for specific edges via owl:Axiom
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

    # 3) write edges
    edges_written = 0
    with open(out_tsv, "w", encoding="utf-8") as f:
        for s, p, o in g:
            # keep full graph of resource/object links; skip literals
            if isinstance(o, Literal):
                continue

            ss, pp, oo = str(s), str(p), str(o)
            w = wmap.get((ss, pp, oo), float(default_weight))

            f.write(f"{ent_id(ss)}\t{rel_id(pp)}\t{ent_id(oo)}\t{w}\n")
            edges_written += 1

    # 4) mappings
    with open(out_tsv + ".entities.tsv", "w", encoding="utf-8") as f:
        for iri, idx in sorted(ent2id.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{iri}\n")

    with open(out_tsv + ".relations.tsv", "w", encoding="utf-8") as f:
        for iri, idx in sorted(rel2id.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{iri}\n")

    print(f"Edges written: {edges_written}")
    print(f"Entities: {len(ent2id)}")
    print(f"Relations: {len(rel2id)}")
    print(f"Weighted edges found via owl:Axiom: {len(wmap)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--default-weight", type=float, default=0.0)
    ap.add_argument("--weight-prop", action="append", default=None,
                    help="Repeatable. Defaults to hasScoreInteraction + hasExpressionValue")
    args = ap.parse_args()

    convert_owl_to_weighted_edgelist(
        owl_file=args.inp,
        out_tsv=args.out,
        default_weight=args.default_weight,
        weight_props=args.weight_prop,
    )

if __name__ == "__main__":
    main()
