"""How the outcomes nest under each choice, and what each leaves the team up 3 facing."""

# what each final outcome leaves the team up 3 facing
KINDS = [("tie", "Trailing team ties or gets another shot", "#c2452d"),
         ("cut", "Trailing team cuts it to 1 or 2", "#d99a00"),
         ("safe", "Trailing team comes away with nothing", "#2a78d6")]  # fmt: skip
# how the parser's outcomes nest under each choice; outcomes under 1% are left out
TREE = {
    "defend": [
        ("Shot", [
            ("Made", [("3-pointer", "Made Three", "tie"), ("2-pointer", "Made Two", "cut")]),
            ("Missed", [("Defensive rebound", "Miss, Defensive Rebound", "safe"),
                        ("Offensive rebound", "Miss, Offensive Rebound", "tie"),
                        ("Time runs out", "Late Miss, Time Runs Out", "safe")]),
            ("Fouled on a 3", "Fouled on a Three", "tie"),
        ]),
        ("Turnover", "Turnover", "safe"),
    ],
    "foul": [
        ("Makes both free throws", "Free Throws, 2 of 2", "cut"),
        ("Misses at least one", [("Makes 1 of 2", "Free Throws, 1 of 2", "cut"),
                                 ("Misses both", "Free Throws, 0 of 2", "safe"),
                                 ("Offensive rebound", "Missed Free Throw, Offensive Rebound",
                                  "tie")]),
        ("Turnover", "Turnover", "safe"),
    ],
}  # fmt: skip


def flatten(nodes, leaves, depth):
    """Rows of [depth, label, chance, win after, kind] in tree order; a branch sums its
    children and has no kind."""
    rows = []
    for node in nodes:
        label, child = node[0], node[1]
        if isinstance(child, str):
            r = leaves.loc[child]
            rows.append([depth, label, r.chance, r.leader_wins_after, node[2]])
        else:
            sub = flatten(child, leaves, depth + 1)
            kids = [x for x in sub if x[0] == depth + 1]
            chance = sum(x[2] for x in kids)
            after = sum(x[2] * x[3] for x in kids) / chance
            rows += [[depth, label, chance, after, None], *sub]
    return rows
