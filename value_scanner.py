import io
import math
import requests
import numpy as np
import pandas as pd


BASE_URL = "https://www.football-data.co.uk/mmz4281"

LEAGUES = {
    "Italia Serie A": "I1",
    "Italia Serie B": "I2",

    "Inghilterra Premier League": "E0",
    "Inghilterra Championship": "E1",
    "Inghilterra League One": "E2",
    "Inghilterra League Two": "E3",

    "Spagna La Liga": "SP1",
    "Spagna Segunda": "SP2",

    "Germania Bundesliga": "D1",
    "Germania Bundesliga 2": "D2",

    "Francia Ligue 1": "F1",
    "Francia Ligue 2": "F2",

    "Olanda Eredivisie": "N1",
    "Belgio Jupiler": "B1",
    "Portogallo Liga": "P1",
    "Turchia Super Lig": "T1",
    "Grecia": "G1",

    "Scozia Premiership": "SC0",
    "Scozia Championship": "SC1",
    "Scozia League One": "SC2",
    "Scozia League Two": "SC3",

    "Polonia": "POL",
    "Romania": "ROM",
    "Russia": "RUS",
    "USA MLS": "USA",
    "Giappone": "JPN",
    "Brasile": "BRA",
    "Argentina": "ARG",
    "Messico": "MEX",
}


def current_season():
    today = pd.Timestamp.today()

    if today.month >= 7:
        y1 = today.year
        y2 = today.year + 1
    else:
        y1 = today.year - 1
        y2 = today.year

    return f"{str(y1)[-2:]}{str(y2)[-2:]}"


def previous_season():
    today = pd.Timestamp.today()

    if today.month >= 7:
        y1 = today.year - 1
        y2 = today.year
    else:
        y1 = today.year - 2
        y2 = today.year - 1

    return f"{str(y1)[-2:]}{str(y2)[-2:]}"


def load_league(league_code, season=None):

    if season is None:
        season = current_season()

    url = f"{BASE_URL}/{season}/{league_code}.csv"

    try:
        r = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        if r.status_code != 200:
            return None

        if len(r.content) < 500:
            return None

        df = pd.read_csv(
            io.BytesIO(r.content),
            encoding="latin1"
        )

        return df

    except Exception:
        return None


def prepare_dataframe(df):

    if df is None or df.empty:
        return None

    df = df.copy()

    if "Date" not in df.columns:
        return None

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        dayfirst=True
    )

    for col in [
        "FTHG",
        "FTAG",
        "HTHG",
        "HTAG",
        "B365H",
        "B365D",
        "B365A",
        "B365>2.5",
        "B365<2.5",
        "BbAvH",
        "BbAvD",
        "BbAvA",
        "BbAv>2.5",
        "BbAv<2.5",
        "MaxH",
        "MaxD",
        "MaxA",
        "Max>2.5",
        "Max<2.5",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


def poisson(k, lam):

    if lam <= 0:
        return 0.0

    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def score_matrix(home_xg, away_xg, max_goals=8):

    matrix = np.zeros(
        (max_goals + 1, max_goals + 1)
    )

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):

            matrix[h, a] = (
                poisson(h, home_xg)
                * poisson(a, away_xg)
            )

    total = matrix.sum()

    if total > 0:
        matrix /= total

    return matrix


def model_probabilities(home_xg, away_xg):

    matrix = score_matrix(
        home_xg,
        away_xg
    )

    p_home = np.tril(matrix, -1).sum()
    p_draw = np.trace(matrix)
    p_away = np.triu(matrix, 1).sum()

    p_over = {}

    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:

        probability = 0

        for h in range(matrix.shape[0]):
            for a in range(matrix.shape[1]):

                if h + a > line:
                    probability += matrix[h, a]

        p_over[line] = probability

    return {
        "home": p_home,
        "draw": p_draw,
        "away": p_away,
        "over": p_over,
    }


def calculate_team_strengths(df):

    played = df.dropna(
        subset=["FTHG", "FTAG"]
    ).copy()

    if played.empty:
        return {}, {}, 0, 0

    league_home_avg = played["FTHG"].mean()
    league_away_avg = played["FTAG"].mean()

    teams = set(
        played["HomeTeam"].dropna()
    ).union(
        set(played["AwayTeam"].dropna())
    )

    stats = {}

    for team in teams:

        home = played[
            played["HomeTeam"] == team
        ]

        away = played[
            played["AwayTeam"] == team
        ]

        home_games = len(home)
        away_games = len(away)

        home_scored = home["FTHG"].sum()
        home_conceded = home["FTAG"].sum()

        away_scored = away["FTAG"].sum()
        away_conceded = away["FTHG"].sum()

        total_games = home_games + away_games

        if total_games == 0:
            continue

        scored = (
            home_scored + away_scored
        ) / total_games

        conceded = (
            home_conceded + away_conceded
        ) / total_games

        home_scored_avg = (
            home_scored / home_games
            if home_games > 0
            else league_home_avg
        )

        home_conceded_avg = (
            home_conceded / home_games
            if home_games > 0
            else league_away_avg
        )

        away_scored_avg = (
            away_scored / away_games
            if away_games > 0
            else league_away_avg
        )

        away_conceded_avg = (
            away_conceded / away_games
            if away_games > 0
            else league_home_avg
        )

        stats[team] = {
            "games": total_games,
            "scored": scored,
            "conceded": conceded,

            "home_scored": home_scored_avg,
            "home_conceded": home_conceded_avg,

            "away_scored": away_scored_avg,
            "away_conceded": away_conceded_avg,
        }

    return (
        stats,
        played,
        league_home_avg,
        league_away_avg
    )


def calculate_xg(
    home_team,
    away_team,
    stats,
    league_home_avg,
    league_away_avg
):

    if home_team not in stats:
        return None, None

    if away_team not in stats:
        return None, None

    home = stats[home_team]
    away = stats[away_team]

    if home["games"] < 2:
        return None, None

    if away["games"] < 2:
        return None, None

    home_attack = (
        home["home_scored"]
        / league_home_avg
        if league_home_avg > 0
        else 1
    )

    home_defense = (
        home["home_conceded"]
        / league_away_avg
        if league_away_avg > 0
        else 1
    )

    away_attack = (
        away["away_scored"]
        / league_away_avg
        if league_away_avg > 0
        else 1
    )

    away_defense = (
        away["away_conceded"]
        / league_home_avg
        if league_home_avg > 0
        else 1
    )

    home_xg = (
        league_home_avg
        * home_attack
        * away_defense
    )

    away_xg = (
        league_away_avg
        * away_attack
        * home_defense
    )

    home_xg = max(0.05, min(home_xg, 5))
    away_xg = max(0.05, min(away_xg, 5))

    return home_xg, away_xg


def get_odds(row):

    odds = {}

    # 1X2
    for name, candidates in {
        "1": [
            "BbAvH",
            "MaxH",
            "B365H"
        ],
        "X": [
            "BbAvD",
            "MaxD",
            "B365D"
        ],
        "2": [
            "BbAvA",
            "MaxA",
            "B365A"
        ],
        "Over 2.5": [
            "BbAv>2.5",
            "Max>2.5",
            "B365>2.5"
        ],
        "Under 2.5": [
            "BbAv<2.5",
            "Max<2.5",
            "B365<2.5"
        ],
    }.items():

        value = None

        for col in candidates:

            if col in row.index:

                try:
                    x = float(row[col])

                    if x > 1:
                        value = x
                        break

                except Exception:
                    pass

        if value is not None:
            odds[name] = value

    return odds


def build_value_row(
    league,
    row,
    stats,
    league_home_avg,
    league_away_avg
):

    home_team = row["HomeTeam"]
    away_team = row["AwayTeam"]

    home_xg, away_xg = calculate_xg(
        home_team,
        away_team,
        stats,
        league_home_avg,
        league_away_avg
    )

    if home_xg is None:
        return []

    probs = model_probabilities(
        home_xg,
        away_xg
    )

    odds = get_odds(row)

    results = []

    markets = {
        "1": probs["home"],
        "X": probs["draw"],
        "2": probs["away"],
        "Over 2.5": probs["over"][2.5],
        "Under 2.5": 1 - probs["over"][2.5],
    }

    for market, probability in markets.items():

        if market not in odds:
            continue

        odd = odds[market]

        if probability <= 0:
            continue

        fair_odd = 1 / probability

        value = (
            probability * odd
        ) - 1

        results.append({

            "Campionato": league,

            "Data": row["Date"].strftime(
                "%d/%m/%Y"
            ),

            "Partita":
                f"{home_team} - {away_team}",

            "Mercato": market,

            "Probabilità": probability,

            "Quota": odd,

            "Quota equa": fair_odd,

            "Value": value,

            "xG casa": home_xg,

            "xG ospite": away_xg,

            "Affidabilità":
                min(
                    100,
                    (
                        stats[home_team]["games"]
                        +
                        stats[away_team]["games"]
                    )
                    / 20
                    * 100
                )
        })

    return results


def scan_all_leagues(
    min_value=0.03,
    min_probability=0.55
):

    all_results = []

    season = current_season()

    for league, code in LEAGUES.items():

        df = load_league(
            code,
            season
        )

        if df is None:
            continue

        df = prepare_dataframe(df)

        if df is None:
            continue

        stats, played, league_home_avg, league_away_avg = \
            calculate_team_strengths(df)

        if not stats:
            continue

        # Solo partite non ancora giocate
        upcoming = df[
            df["FTHG"].isna()
            |
            df["FTAG"].isna()
        ].copy()

        if upcoming.empty:
            continue

        for _, row in upcoming.iterrows():

            try:

                results = build_value_row(
                    league,
                    row,
                    stats,
                    league_home_avg,
                    league_away_avg
                )

                for result in results:

                    if (
                        result["Value"]
                        >= min_value
                        and
                        result["Probabilità"]
                        >= min_probability
                    ):
                        all_results.append(result)

            except Exception:
                continue

    if not all_results:
        return pd.DataFrame()

    result_df = pd.DataFrame(
        all_results
    )

    result_df = result_df.sort_values(
        "Value",
        ascending=False
    )

    return result_df.reset_index(
        drop=True
    )
