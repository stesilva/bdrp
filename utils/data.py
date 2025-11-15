
import os

S = os.sep

def locate_file(filepath):
    """ Locate file relative to project root directory """
    directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return directory + '/' + filepath



def load_strings(file):
    """ Read triples from file keeping only first three columns (s, p, o) """
    with open(file, 'r') as f:
        return [line.split()[:3] for line in f]




def load_link_prediction_data(name, use_test_set=False, limit=None):
    """
    Load knowledge graphs for link Prediction  experiment.

    Source: https://github.com/pbloem/gated-rgcn/blob/1bde7f28af8028f468349b2d760c17d5c908b58b/kgmodels/data.py#L218

    :param name: Dataset name ('cn15k')
    :param use_test_set: If true, load the canonical test set, otherwise load validation set from file.
    :param limit: If set, only the first n triples are used.
    :return: Link prediction test and train sets:
              - (n2i, nodes): node-to-index and index-to-node mappings
              - (r2i, relations): relation-to-index and index-to-relation mappings
              - train: list of edges [subject, predicate object]
              - test: list of edges [subject, predicate object]
              - all_triples: sets of tuples (subject, predicate object)
    """

    if name.lower() == 'cn15k':
        train_file = locate_file('nl/train.tsv')
        val_file = locate_file('cn15k/val.tsv')
        test_file = locate_file('cn15k/test.tsv')
    elif name.lower() == 'nl27k':
        train_file = locate_file('nl27k/train.tsv')
        val_file = locate_file('nl27k/val.tsv')
        test_file = locate_file('nl27k/test.tsv')
    else:
        raise ValueError(f'Could not find \'{name}\' dataset')

    train = load_strings(train_file)
    val = load_strings(val_file)
    test = load_strings(test_file)

    if not use_test_set:
        test = val
    # else:
    #     train = train + val

    if limit:
        train = train[:limit]
        test = test[:limit]

    # Mappings for nodes (n) and relations (r)
    nodes, rels = set(), set()
    for s, p, o in train + val + test:
        nodes.add(s)
        rels.add(p)
        nodes.add(o)

    n, r = list(nodes), list(rels)
    n2i, r2i = {n: i for i, n in enumerate(nodes)}, {r: i for i, r in enumerate(rels)}

    all_triples = set()
    for s, p, o in train + val + test:
        all_triples.add((n2i[s], r2i[p], n2i[o]))

    train = [[n2i[st[0]], r2i[st[1]], n2i[st[2]]] for st in train]
    test = [[n2i[st[0]], r2i[st[1]], n2i[st[2]]] for st in test]

    return (n2i, n), (r2i, r), train, test, all_triples
