from draft_value.data import parse_page

PAGE = """
<table id="stats"><thead>
<tr><th></th><th colspan="3">Round 1</th><th colspan="3">Totals</th>
<th colspan="2">Per Game</th></tr>
<tr><th>Rk</th><th>Pk</th><th>Tm</th><th>Player</th><th>G</th><th>WS</th><th>VORP</th>
<th>MP</th><th>MP</th></tr>
</thead><tbody>
<tr><th>1</th><td>1</td><td>PHI</td><td>Ben Simmons</td><td>383</td><td>35.4</td><td>15.0</td>
<td>11910</td><td>31.1</td></tr>
<tr class="over_header thead"><th></th><th colspan="3">Round 2</th><th colspan="3"></th>
<th colspan="2"></th></tr>
<tr class="thead"><th>Rk</th><th>Pk</th><th>Tm</th><th>Player</th><th>G</th><th>WS</th><th>VORP</th>
<th>MP</th><th>MP</th></tr>
<tr><th>2</th><td>31</td><td>BOS</td><td>Never Played</td><td></td><td></td><td></td>
<td></td><td></td></tr>
</tbody></table>
"""


def test_parse_page_keeps_draftees_reads_rounds_and_zeroes_players_who_never_played():
    rows = parse_page(PAGE, 2016)

    assert rows["pick"].tolist() == [1, 31]  # the repeated header row is gone
    assert rows["player"].tolist() == ["Ben Simmons", "Never Played"]
    assert rows["year"].unique().tolist() == [2016]
    assert rows["round"].tolist() == [1, 2]  # from the "Round 2" header row
    assert rows.loc[0, "win_shares"] == 35.4
    assert rows.loc[1, ["games", "win_shares", "vorp"]].tolist() == [0, 0.0, 0.0]
