import math
import re
from datetime import datetime, date
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# CONFIGURAZIONE
# ============================================================

st.set_page_config(
    page_title="Football Analyzer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_URL = "https://www.football-data.co.uk/mmz4281"
FIXTURES_URL = "https://football-data.co.uk/matches/resources/fixtures.csv"


# ============================================================
# STILE
# ============================================================

st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .block-container {
        padding-top: 1.5rem;
    }

    .metric-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 10px;
    }

    .value-positive {
        font-weight: bold;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CAMPIONATI
# ============================================================

LEAGUES = {
    "Italia": {
        "I1": "Serie A",
        "I2": "Serie B",
    },
    "Inghilterra": {
        "E0": "Premier League",
        "E1": "Championship",
        "E2": "League One",
        "E3": "League Two",
    },
    "Spagna": {
        "SP1": "La Liga",
        "SP2": "Segunda Division",
    },
    "Germania": {
        "D1": "Bundesliga",
        "D2": "2. Bundesliga",
    },
    "Francia": {
        "F1": "Ligue 1",
        "F2": "Ligue 2",
    },
    "Olanda": {
        "N1": "Eredivisie",
    },
    "Belgio": {
        "B1": "Belgian First Division",
    },
    "Portogallo": {
        "P1": "Primeira Liga",
    },
    "Turchia": {
        "T1": "Super Lig",
    },
    "Grecia": {
        "G1": "Super League",
    },
    "Scozia": {
        "SC0": "Premiership",
        "SC1": "Championship",
        "SC2": "League One",
        "SC3": "League Two",
    },
    "Polonia": {
        "POL": "Poland Ekstraklasa",
    },
    "Romania": {
        "ROM": "Romania Liga 1",
    },
    "Russia": {
        "RUS": "Russia Premier League",
    },
    "USA": {
        "USA": "MLS",
    },
    "Giappone": {
        "JPN": "J League",
    },
    "Brasile": {
        "BRA": "Brazil Serie A",
    },
    "Argentina": {
        "ARG": "Argentina Primera Division",
    },
    "Messico": {
        "MEX": "Mexico Liga MX",
    },
}


# ============================================================
# UTILITY
# ============================================================

def current_season():
    """
    Restituisce la stagione corrente nel formato Football-Data.
    Esempio:
    2026/27 -> 2627
    """
    today = date.today()

    if today.month >= 7:
        y1 = today.year
        y2 = today.year + 1
    else:
        y1 = today.year - 1
        y2 = today.year

    return f"{str(y1)[-2:]}{str(y2)[-2:]}"


def previous_season():
    code = current_season()

    y1 = int(code[:2])
    y2 = int(code[2:])

    return f"{(y1 - 1) % 100:02d}{(y2 - 1) % 100:02d}"


def season_label(code):
    return f"20{code[:2]}/20{code[2:]}"


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def poisson_probability(lmbda, k):
    if lmbda < 0:
        return 0.0

    try:
        return math.exp(-lmbda) * (lmbda ** k) / math.factorial(k)
    except Exception:
        return 0.0


def fair_odds(probability):
    if probability is None or probability <= 0:
        return np.nan

    return 1.0 / probability


def calculate_value(probability, odds):
    if probability is None or odds is None:
        return np.nan

    if probability <= 0 or odds <= 0:
        return np.nan

    return probability * odds - 1.0


# ============================================================
# DOWNLOAD DATI
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def download_csv(url):
    try:
        response = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        )

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}"

        content = response.content.decode(
            "latin1",
            errors="replace",
        )

        df = pd.read_csv(
            StringIO(content),
            low_memory=False,
        )

        return df, None

    except Exception as exc:
        return None, str(exc)


@st.cache_data(ttl=3600, show_spinner=False)
def load_league(league_code, season_code):
    url = f"{BASE_URL}/{season_code}/{league_code}.csv"

    df, error = download_csv(url)

    if df is None:
        return None, error, url

    return df, None, url


@st.cache_data(ttl=1800, show_spinner=False)
def load_fixtures():
    df, error = download_csv(FIXTURES_URL)

    if df is None:
        return None, error

    return df, None


# ============================================================
# PREPARAZIONE DATAFRAME
# ============================================================

def prepare_dataframe(df):
    if df is None or df.empty:
        return None

    df = df.copy()

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    required = [
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
    ]

    for col in required:
        if col not in df.columns:
            return None

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce",
            dayfirst=True,
        )

    for col in [
        "FTHG",
        "FTAG",
        "HTHG",
        "HTAG",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    df = df.dropna(
        subset=[
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
        ]
    )

    if "Date" in df.columns:
        df = df.sort_values("Date")

    return df.reset_index(drop=True)


# ============================================================
# STATISTICHE SQUADRA
# ============================================================

def team_matches(df, team, last_n=None):
    if df is None or df.empty:
        return pd.DataFrame()

    home = df[df["HomeTeam"] == team].copy()
    away = df[df["AwayTeam"] == team].copy()

    home["GF"] = home["FTHG"]
    home["GA"] = home["FTAG"]
    home["Venue"] = "Casa"
    home["Opponent"] = home["AwayTeam"]

    away["GF"] = away["FTAG"]
    away["GA"] = away["FTHG"]
    away["Venue"] = "Trasferta"
    away["Opponent"] = away["HomeTeam"]

    result = pd.concat(
        [home, away],
        ignore_index=True,
    )

    if "Date" in result.columns:
        result = result.sort_values("Date")

    if last_n:
        result = result.tail(last_n)

    return result


def team_stats(df, team, last_n=None):
    matches = team_matches(
        df,
        team,
        last_n,
    )

    if matches.empty:
        return {
            "matches": 0,
            "gf": 0,
            "ga": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "avg_gf": 0,
            "avg_ga": 0,
            "over05": 0,
            "over15": 0,
            "over25": 0,
            "over35": 0,
            "over45": 0,
            "btts": 0,
            "clean_sheet": 0,
            "failed_to_score": 0,
        }

    gf = matches["GF"].astype(float)
    ga = matches["GA"].astype(float)

    wins = int((gf > ga).sum())
    draws = int((gf == ga).sum())
    losses = int((gf < ga).sum())

    total_goals = gf + ga

    stats = {
        "matches": len(matches),
        "gf": gf.sum(),
        "ga": ga.sum(),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "avg_gf": gf.mean(),
        "avg_ga": ga.mean(),
        "over05": (total_goals > 0).mean(),
        "over15": (total_goals > 1).mean(),
        "over25": (total_goals > 2).mean(),
        "over35": (total_goals > 3).mean(),
        "over45": (total_goals > 4).mean(),
        "btts": ((gf > 0) & (ga > 0)).mean(),
        "clean_sheet": (ga == 0).mean(),
        "failed_to_score": (gf == 0).mean(),
    }

    return stats


# ============================================================
# STATISTICHE CAMPIONATO
# ============================================================

def league_averages(df):
    if df is None or df.empty:
        return {
            "home_goals": 1.35,
            "away_goals": 1.10,
            "total_goals": 2.45,
        }

    home_goals = df["FTHG"].mean()
    away_goals = df["FTAG"].mean()

    return {
        "home_goals": float(home_goals),
        "away_goals": float(away_goals),
        "total_goals": float(home_goals + away_goals),
    }


# ============================================================
# FORZA SQUADRE
# ============================================================

def calculate_team_strengths(df, home_team, away_team):
    averages = league_averages(df)

    home_history = team_stats(
        df,
        home_team,
        last_n=15,
    )

    away_history = team_stats(
        df,
        away_team,
        last_n=15,
    )

    league_home = averages["home_goals"]
    league_away = averages["away_goals"]

    if league_home <= 0:
        league_home = 1.35

    if league_away <= 0:
        league_away = 1.10

    home_attack = home_history["avg_gf"] / league_home
    home_defense = home_history["avg_ga"] / league_away

    away_attack = away_history["avg_gf"] / league_away
    away_defense = away_history["avg_ga"] / league_home

    home_xg = league_home * home_attack * away_defense
    away_xg = league_away * away_attack * home_defense

    home_xg = max(0.05, min(home_xg, 5.0))
    away_xg = max(0.05, min(away_xg, 5.0))

    return {
        "home_xg": home_xg,
        "away_xg": away_xg,
        "home_attack": home_attack,
        "home_defense": home_defense,
        "away_attack": away_attack,
        "away_defense": away_defense,
        "home_stats": home_history,
        "away_stats": away_history,
    }


# ============================================================
# MODELLO POISSON
# ============================================================

def score_matrix(home_xg, away_xg, max_goals=8):
    matrix = np.zeros(
        (max_goals + 1, max_goals + 1)
    )

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):

            matrix[
                home_goals,
                away_goals
            ] = (
                poisson_probability(
                    home_xg,
                    home_goals,
                )
                *
                poisson_probability(
                    away_xg,
                    away_goals,
                )
            )

    total = matrix.sum()

    if total > 0:
        matrix /= total

    return matrix


def calculate_probabilities(home_xg, away_xg):
    matrix = score_matrix(
        home_xg,
        away_xg,
    )

    home_win = np.tril(
        matrix,
        -1,
    ).sum()

    draw = np.trace(matrix)

    away_win = np.triu(
        matrix,
        1,
    ).sum()

    total_xg = home_xg + away_xg

    over05 = 1 - poisson_probability(
        total_xg,
        0,
    )

    under05 = 1 - over05

    over15 = 1 - (
        poisson_probability(total_xg, 0)
        +
        poisson_probability(total_xg, 1)
    )

    under15 = 1 - over15

    under25 = sum(
        poisson_probability(total_xg, i)
        for i in range(3)
    )

    over25 = 1 - under25

    under35 = sum(
        poisson_probability(total_xg, i)
        for i in range(4)
    )

    over35 = 1 - under35

    under45 = sum(
        poisson_probability(total_xg, i)
        for i in range(5)
    )

    over45 = 1 - under45

    home_scores = 1 - math.exp(-home_xg)
    away_scores = 1 - math.exp(-away_xg)

    home_clean = math.exp(-away_xg)
    away_clean = math.exp(-home_xg)

    home_no_goal = math.exp(-home_xg)
    away_no_goal = math.exp(-away_xg)

    btts = (
        1
        - home_no_goal
        - away_no_goal
        + math.exp(-total_xg)
    )

    return {
        "1": home_win,
        "X": draw,
        "2": away_win,

        "1X": home_win + draw,
        "X2": draw + away_win,
        "12": home_win + away_win,

        "Over 0.5": over05,
        "Under 0.5": under05,

        "Over 1.5": over15,
        "Under 1.5": under15,

        "Over 2.5": over25,
        "Under 2.5": under25,

        "Over 3.5": over35,
        "Under 3.5": under35,

        "Over 4.5": over45,
        "Under 4.5": under45,

        "Goal": btts,
        "No Goal": 1 - btts,

        "Casa segna": home_scores,
        "Casa non segna": home_no_goal,

        "Trasferta segna": away_scores,
        "Trasferta non segna": away_no_goal,

        "Clean Sheet Casa": home_clean,
        "Clean Sheet Trasferta": away_clean,
    }


# ============================================================
# RISULTATI ESATTI
# ============================================================

def top_exact_scores(matrix, number=10):
    results = []

    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):

            results.append(
                {
                    "Risultato": f"{h}-{a}",
                    "Probabilità": matrix[h, a],
                }
            )

    results.sort(
        key=lambda x: x["Probabilità"],
        reverse=True,
    )

    return results[:number]


# ============================================================
# NORMALIZZAZIONE NOMI SQUADRE
# ============================================================

def normalize_team_name(name):
    if pd.isna(name):
        return ""

    name = str(name).lower().strip()

    replacements = {
        "man utd": "manchester united",
        "man united": "manchester united",
        "man city": "manchester city",
        "spurs": "tottenham",
        "wolves": "wolverhampton",
        "newcastle utd": "newcastle",
        "west ham utd": "west ham",
        "nott'm forest": "nottingham forest",
        "psv eindhoven": "psv",
        "inter": "internazionale",
        "inter milan": "internazionale",
        "milan": "milan",
        "roma": "roma",
    }

    name = replacements.get(
        name,
        name,
    )

    name = re.sub(
        r"[^a-z0-9 ]",
        "",
        name,
    )

    name = re.sub(
        r"\s+",
        " ",
        name,
    )

    return name


# ============================================================
# ODDS
# ============================================================

def extract_odds(row):
    """
    Cerca le quote 1X2 e Over/Under presenti nel dataset.
    Football-Data usa colonne differenti a seconda del bookmaker.
    """

    result = {
        "home_odds": np.nan,
        "draw_odds": np.nan,
        "away_odds": np.nan,
        "over25_odds": np.nan,
        "under25_odds": np.nan,
    }

    if row is None:
        return result

    # 1X2
    for col in [
        "B365H",
        "AvgH",
        "PSH",
        "WHH",
        "VCH",
        "MaxH",
    ]:
        if col in row.index:
            value = safe_float(row[col])
            if not np.isnan(value) and value > 1:
                result["home_odds"] = value
                break

    for col in [
        "B365D",
        "AvgD",
        "PSD",
        "WHD",
        "VCD",
        "MaxD",
    ]:
        if col in row.index:
            value = safe_float(row[col])
            if not np.isnan(value) and value > 1:
                result["draw_odds"] = value
                break

    for col in [
        "B365A",
        "AvgA",
        "PSA",
        "WHA",
        "VCA",
        "MaxA",
    ]:
        if col in row.index:
            value = safe_float(row[col])
            if not np.isnan(value) and value > 1:
                result["away_odds"] = value
                break

    # Over / Under 2.5
    for col in [
        "B365>2.5",
        "Avg>2.5",
        "P>2.5",
        "Max>2.5",
    ]:
        if col in row.index:
            value = safe_float(row[col])
            if not np.isnan(value) and value > 1:
                result["over25_odds"] = value
                break

    for col in [
        "B365<2.5",
        "Avg<2.5",
        "P<2.5",
        "Max<2.5",
    ]:
        if col in row.index:
            value = safe_float(row[col])
            if not np.isnan(value) and value > 1:
                result["under25_odds"] = value
                break

    return result


# ============================================================
# VALUE SCANNER
# ============================================================

def calculate_value_row(
    league_name,
    league_code,
    row,
    history_df,
):
    home = str(row["HomeTeam"])
    away = str(row["AwayTeam"])

    try:
        strengths = calculate_team_strengths(
            history_df,
            home,
            away,
        )

        probabilities = calculate_probabilities(
            strengths["home_xg"],
            strengths["away_xg"],
        )

        odds = extract_odds(row)

        candidates = []

        markets = [
            (
                "1",
                probabilities["1"],
                odds["home_odds"],
            ),
            (
                "X",
                probabilities["X"],
                odds["draw_odds"],
            ),
            (
                "2",
                probabilities["2"],
                odds["away_odds"],
            ),
            (
                "Over 2.5",
                probabilities["Over 2.5"],
                odds["over25_odds"],
            ),
            (
                "Under 2.5",
                probabilities["Under 2.5"],
                odds["under25_odds"],
            ),
        ]

        for market, probability, odd in markets:

            if pd.isna(odd):
                continue

            value = calculate_value(
                probability,
                odd,
            )

            if pd.isna(value):
                continue

            candidates.append(
                {
                    "Campionato": league_name,
                    "Codice": league_code,
                    "Partita": f"{home} - {away}",
                    "Casa": home,
                    "Trasferta": away,
                    "Mercato": market,
                    "Probabilità": probability,
                    "Quota": odd,
                    "Quota equa": fair_odds(probability),
                    "Value": value,
                    "xG Casa": strengths["home_xg"],
                    "xG Trasferta": strengths["away_xg"],
                }
            )

        if not candidates:
            return []

        return candidates

    except Exception:
        return []


def scan_league(
    league_code,
    league_name,
    season_code,
):
    df, error, url = load_league(
        league_code,
        season_code,
    )

    if df is None:
        return [], {
            "league": league_name,
            "code": league_code,
            "status": "ERRORE",
            "error": error,
        }

    df = prepare_dataframe(df)

    if df is None or df.empty:
        return [], {
            "league": league_name,
            "code": league_code,
            "status": "VUOTO",
            "error": "Dataset vuoto",
        }

    # Per lo scanner utilizziamo solo partite già presenti
    # nel dataset storico.
    rows = []

    # Limitiamo la scansione alle ultime partite disponibili
    # per evitare tempi inutilmente elevati.
    recent = df.tail(150)

    for _, row in recent.iterrows():

        candidates = calculate_value_row(
            league_name,
            league_code,
            row,
            df,
        )

        rows.extend(candidates)

    return rows, {
        "league": league_name,
        "code": league_code,
        "status": "OK",
        "matches": len(df),
        "scanned": len(recent),
        "url": url,
    }


def scan_all_leagues(
    selected_countries,
    season_code,
    min_value=0.03,
):
    all_results = []
    debug = []

    for country in selected_countries:

        league_dict = LEAGUES.get(
            country,
            {},
        )

        for code, league_name in league_dict.items():

            results, info = scan_league(
                code,
                league_name,
                season_code,
            )

            debug.append(info)

            for result in results:

                if result["Value"] >= min_value:
                    all_results.append(result)

    if all_results:
        result_df = pd.DataFrame(
            all_results
        )

        result_df = result_df.sort_values(
            "Value",
            ascending=False,
        )

    else:
        result_df = pd.DataFrame()

    return result_df, debug


# ============================================================
# FIXTURES FUTURI
# ============================================================

def prepare_fixtures(df):
    if df is None or df.empty:
        return None

    df = df.copy()

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    # Tentativi per identificare data
    date_col = None

    for col in [
        "Date",
        "date",
        "MatchDate",
        "match_date",
    ]:
        if col in df.columns:
            date_col = col
            break

    if date_col:
        df["FixtureDate"] = pd.to_datetime(
            df[date_col],
            errors="coerce",
            dayfirst=True,
        )
    else:
        df["FixtureDate"] = pd.NaT

    if "HomeTeam" not in df.columns:
        return None

    if "AwayTeam" not in df.columns:
        return None

    return df


# ============================================================
# VISUALIZZAZIONE PROBABILITA'
# ============================================================

def probability_table(probabilities):
    rows = []

    for market, probability in probabilities.items():

        rows.append(
            {
                "Mercato": market,
                "Probabilità": probability,
                "Quota equa": fair_odds(
                    probability
                ),
            }
        )

    df = pd.DataFrame(rows)

    df["Probabilità"] = (
        df["Probabilità"] * 100
    ).round(2)

    df["Quota equa"] = df["Quota equa"].round(2)

    return df


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Football Analyzer")

page = st.sidebar.radio(
    "Sezione",
    [
        "Dashboard",
        "Analisi partita",
        "Scanner",
        "Debug dati",
    ],
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Modello statistico basato su dati Football-Data."
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("⚽ Football Analyzer")

    st.write(
        "Analisi statistica delle partite di calcio "
        "con modello Poisson e confronto probabilità/quote."
    )

    st.markdown("---")

    st.subheader("Campionati disponibili")

    rows = []

    for country, leagues in LEAGUES.items():

        for code, name in leagues.items():

            rows.append(
                {
                    "Paese": country,
                    "Codice": code,
                    "Campionato": name,
                }
            )

    leagues_df = pd.DataFrame(rows)

    st.dataframe(
        leagues_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Campionati",
            len(rows),
        )

    with col2:
        st.metric(
            "Stagione",
            season_label(
                current_season()
            ),
        )

    with col3:
        st.metric(
            "Modello",
            "Poisson",
        )

    st.info(
        "Il Value Scanner usa i dati disponibili nei dataset "
        "Football-Data. Le quote storiche presenti nei CSV "
        "non devono essere confuse con quote live."
    )


# ============================================================
# ANALISI PARTITA
# ============================================================

elif page == "Analisi partita":

    st.title("📊 Analisi partita")

    col1, col2 = st.columns(2)

    with col1:

        country = st.selectbox(
            "Paese",
            list(LEAGUES.keys()),
        )

        league_items = LEAGUES[country]

        league_code = st.selectbox(
            "Campionato",
            list(league_items.keys()),
            format_func=lambda x: league_items[x],
        )

    season_code = st.selectbox(
        "Stagione",
        [
            current_season(),
            previous_season(),
            "2425",
            "2324",
            "2223",
        ],
        format_func=season_label,
    )

    df, error, url = load_league(
        league_code,
        season_code,
    )

    if df is None:

        st.error(
            f"Impossibile scaricare il campionato: {error}"
        )

        st.stop()

    df = prepare_dataframe(df)

    if df is None or df.empty:

        st.error(
            "Non sono disponibili dati validi."
        )

        st.stop()

    teams = sorted(
        set(df["HomeTeam"].dropna())
        |
        set(df["AwayTeam"].dropna())
    )

    col1, col2 = st.columns(2)

    with col1:

        home_team = st.selectbox(
            "Squadra casa",
            teams,
        )

    with col2:

        away_options = [
            x for x in teams
            if x != home_team
        ]

        away_team = st.selectbox(
            "Squadra ospite",
            away_options,
        )

    if st.button(
        "ANALIZZA PARTITA",
        type="primary",
        use_container_width=True,
    ):

        strengths = calculate_team_strengths(
            df,
            home_team,
            away_team,
        )

        probabilities = calculate_probabilities(
            strengths["home_xg"],
            strengths["away_xg"],
        )

        matrix = score_matrix(
            strengths["home_xg"],
            strengths["away_xg"],
        )

        st.markdown("---")

        st.subheader(
            f"{home_team} - {away_team}"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "xG Casa",
                f"{strengths['home_xg']:.2f}",
            )

        with c2:
            st.metric(
                "xG Trasferta",
                f"{strengths['away_xg']:.2f}",
            )

        with c3:
            st.metric(
                "xG Totali",
                f"{strengths['home_xg'] + strengths['away_xg']:.2f}",
            )

        st.markdown("---")

        st.subheader("1X2")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "1",
                f"{probabilities['1'] * 100:.1f}%",
            )

        with c2:
            st.metric(
                "X",
                f"{probabilities['X'] * 100:.1f}%",
            )

        with c3:
            st.metric(
                "2",
                f"{probabilities['2'] * 100:.1f}%",
            )

        st.subheader("Doppia chance")

        dc_df = pd.DataFrame(
            {
                "Mercato": [
                    "1X",
                    "X2",
                    "12",
                ],
                "Probabilità": [
                    probabilities["1X"],
                    probabilities["X2"],
                    probabilities["12"],
                ],
            }
        )

        dc_df["Probabilità"] *= 100

        st.dataframe(
            dc_df.style.format(
                {
                    "Probabilità": "{:.1f}%"
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(
            "Over / Under"
        )

        ou_rows = []

        for line in [
            "0.5",
            "1.5",
            "2.5",
            "3.5",
            "4.5",
        ]:

            ou_rows.append(
                {
                    "Linea": line,
                    "Over": probabilities[
                        f"Over {line}"
                    ],
                    "Under": probabilities[
                        f"Under {line}"
                    ],
                }
            )

        ou_df = pd.DataFrame(
            ou_rows
        )

        ou_df["Over"] *= 100
        ou_df["Under"] *= 100

        st.dataframe(
            ou_df.style.format(
                {
                    "Over": "{:.1f}%",
                    "Under": "{:.1f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(
            "Goal / No Goal"
        )

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Goal",
                f"{probabilities['Goal'] * 100:.1f}%",
            )

        with c2:
            st.metric(
                "No Goal",
                f"{probabilities['No Goal'] * 100:.1f}%",
            )

        st.subheader(
            "Team to Score"
        )

        score_df = pd.DataFrame(
            {
                "Mercato": [
                    home_team,
                    away_team,
                ],
                "Segna": [
                    probabilities["Casa segna"],
                    probabilities["Trasferta segna"],
                ],
                "Non segna": [
                    probabilities["Casa non segna"],
                    probabilities["Trasferta non segna"],
                ],
            }
        )

        score_df["Segna"] *= 100
        score_df["Non segna"] *= 100

        st.dataframe(
            score_df.style.format(
                {
                    "Segna": "{:.1f}%",
                    "Non segna": "{:.1f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(
            "Risultati esatti più probabili"
        )

        exact_scores = top_exact_scores(
            matrix,
            10,
        )

        exact_df = pd.DataFrame(
            exact_scores
        )

        exact_df["Probabilità"] *= 100

        st.dataframe(
            exact_df.style.format(
                {
                    "Probabilità": "{:.2f}%"
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        st.subheader(
            "Statistiche squadre"
        )

        home_stats = strengths[
            "home_stats"
        ]

        away_stats = strengths[
            "away_stats"
        ]

        stats_df = pd.DataFrame(
            {
                home_team: {
                    "Partite": home_stats["matches"],
                    "Gol fatti": home_stats["gf"],
                    "Gol subiti": home_stats["ga"],
                    "Vittorie": home_stats["wins"],
                    "Pareggi": home_stats["draws"],
                    "Sconfitte": home_stats["losses"],
                    "Media gol fatti": home_stats["avg_gf"],
                    "Media gol subiti": home_stats["avg_ga"],
                    "Over 2.5": home_stats["over25"],
                    "Goal": home_stats["btts"],
                    "Clean Sheet": home_stats["clean_sheet"],
                },
                away_team: {
                    "Partite": away_stats["matches"],
                    "Gol fatti": away_stats["gf"],
                    "Gol subiti": away_stats["ga"],
                    "Vittorie": away_stats["wins"],
                    "Pareggi": away_stats["draws"],
                    "Sconfitte": away_stats["losses"],
                    "Media gol fatti": away_stats["avg_gf"],
                    "Media gol subiti": away_stats["avg_ga"],
                    "Over 2.5": away_stats["over25"],
                    "Goal": away_stats["btts"],
                    "Clean Sheet": away_stats["clean_sheet"],
                },
            }
        )

        for col in [
            "Over 2.5",
            "Goal",
            "Clean Sheet",
        ]:
            stats_df.loc[col] *= 100

        st.dataframe(
            stats_df.style.format(
                {
                    "Media gol fatti": "{:.2f}",
                    "Media gol subiti": "{:.2f}",
                    "Over 2.5": "{:.1f}%",
                    "Goal": "{:.1f}%",
                    "Clean Sheet": "{:.1f}%",
                }
            ),
            use_container_width=True,
        )

        st.subheader(
            "Ultime partite"
        )

        home_recent = team_matches(
            df,
            home_team,
            10,
        )

        away_recent = team_matches(
            df,
            away_team,
            10,
        )

        c1, c2 = st.columns(2)

        with c1:

            st.write(home_team)

            if not home_recent.empty:

                display = home_recent[
                    [
                        "Date",
                        "Opponent",
                        "Venue",
                        "GF",
                        "GA",
                    ]
                ].copy()

                st.dataframe(
                    display,
                    use_container_width=True,
                    hide_index=True,
                )

        with c2:

            st.write(away_team)

            if not away_recent.empty:

                display = away_recent[
                    [
                        "Date",
                        "Opponent",
                        "Venue",
                        "GF",
                        "GA",
                    ]
                ].copy()

                st.dataframe(
                    display,
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("---")

        st.subheader(
            "Probabilità complete"
        )

        full_probability_df = probability_table(
            probabilities
        )

        st.dataframe(
            full_probability_df.style.format(
                {
                    "Probabilità": "{:.2f}%",
                    "Quota equa": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "Il modello Poisson stima la distribuzione dei gol "
            "partendo dagli xG stimati. Non rappresenta una "
            "garanzia sull'esito della partita."
        )


# ============================================================
# VALUE SCANNER
# ============================================================

elif page == "Value Scanner":

    st.title("🎯 VALUE SCANNER")

    st.write(
        "Ricerca automatica dei mercati nei campionati "
        "selezionati dove la probabilità stimata dal modello "
        "è superiore a quella implicita nella quota."
    )

    st.warning(
        "Il Value è un indicatore matematico del modello. "
        "Non significa che una giocata sia certa o garantita."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        selected_countries = st.multiselect(
            "Campionati da analizzare",
            list(LEAGUES.keys()),
            default=[
                "Italia",
                "Inghilterra",
                "Spagna",
                "Germania",
                "Francia",
            ],
        )

    with col2:

        season_code = st.selectbox(
            "Stagione",
            [
                current_season(),
                previous_season(),
                "2425",
                "2324",
            ],
            format_func=season_label,
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        min_value_percent = st.number_input(
            "Value minimo %",
            min_value=-50.0,
            max_value=200.0,
            value=3.0,
            step=1.0,
        )

    with col2:

        min_probability_percent = st.number_input(
            "Probabilità minima %",
            min_value=0.0,
            max_value=100.0,
            value=55.0,
            step=5.0,
        )

    with col3:

        max_results = st.number_input(
            "Numero risultati",
            min_value=5,
            max_value=200,
            value=30,
            step=5,
        )

    if st.button(
        "AVVIA VALUE SCANNER",
        type="primary",
        use_container_width=True,
    ):

        if not selected_countries:

            st.error(
                "Seleziona almeno un campionato."
            )

            st.stop()

        with st.spinner(
            "Analisi dei campionati in corso..."
        ):

            result_df, debug = scan_all_leagues(
                selected_countries,
                season_code,
                min_value=min_value_percent / 100,
            )

        st.session_state[
            "scanner_results"
        ] = result_df

        st.session_state[
            "scanner_debug"
        ] = debug

    result_df = st.session_state.get(
        "scanner_results",
        pd.DataFrame(),
    )

    if not result_df.empty:

        result_df = result_df[
            result_df["Probabilità"]
            >= min_probability_percent / 100
        ].copy()

        result_df = result_df.sort_values(
            "Value",
            ascending=False,
        ).head(
            int(max_results)
        )

        st.markdown("---")

        st.subheader(
            f"Risultati trovati: {len(result_df)}"
        )

        display_df = result_df[
            [
                "Campionato",
                "Partita",
                "Mercato",
                "Probabilità",
                "Quota",
                "Quota equa",
                "Value",
                "xG Casa",
                "xG Trasferta",
            ]
        ].copy()

        display_df[
            "Probabilità"
        ] *= 100

        display_df[
            "Value"
        ] *= 100

        display_df[
            "Probabilità"
        ] = display_df[
            "Probabilità"
        ].round(1)

        display_df[
            "Value"
        ] = display_df[
            "Value"
        ].round(1)

        display_df[
            "Quota"
        ] = display_df[
            "Quota"
        ].round(2)

        display_df[
            "Quota equa"
        ] = display_df[
            "Quota equa"
        ].round(2)

        display_df[
            "xG Casa"
        ] = display_df[
            "xG Casa"
        ].round(2)

        display_df[
            "xG Trasferta"
        ] = display_df[
            "xG Trasferta"
        ].round(2)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        st.subheader(
            "Migliori opportunità secondo il modello"
        )

        for _, item in result_df.head(10).iterrows():

            probability = item[
                "Probabilità"
            ] * 100

            value = item[
                "Value"
            ] * 100

            st.markdown(
                f"""
                **{item['Partita']}**

                Campionato: {item['Campionato']}  
                Mercato: **{item['Mercato']}**  
                Probabilità modello: **{probability:.1f}%**  
                Quota: **{item['Quota']:.2f}**  
                Quota equa: **{item['Quota equa']:.2f}**  
                Value: **{value:+.1f}%**  
                xG: **{item['xG Casa']:.2f} - {item['xG Trasferta']:.2f}**
                """
            )

            st.markdown("---")

    elif "scanner_results" in st.session_state:

        st.info(
            "Nessun risultato soddisfa i filtri impostati."
        )


# ============================================================
# DEBUG DATI
# ============================================================

elif page == "Debug dati":

    st.title("🔎 Controllo dati")

    st.write(
        "Questa sezione serve per verificare quali campionati "
        "sono effettivamente disponibili e quali dataset "
        "vengono scaricati."
    )

    season_code = st.selectbox(
        "Stagione da verificare",
        [
            current_season(),
            previous_season(),
            "2425",
            "2324",
        ],
        format_func=season_label,
    )

    if st.button(
        "CONTROLLA TUTTI I CAMPIONATI",
        type="primary",
        use_container_width=True,
    ):

        debug_rows = []

        progress = st.progress(0)

        all_leagues = []

        for country, leagues in LEAGUES.items():

            for code, name in leagues.items():

                all_leagues.append(
                    (
                        country,
                        code,
                        name,
                    )
                )

        total = len(all_leagues)

        for index, (
            country,
            code,
            name,
        ) in enumerate(all_leagues):

            df, error, url = load_league(
                code,
                season_code,
            )

            if df is None:

                debug_rows.append(
                    {
                        "Paese": country,
                        "Codice": code,
                        "Campionato": name,
                        "Stato": "ERRORE",
                        "Partite": 0,
                        "Errore": error,
                    }
                )

            else:

                prepared = prepare_dataframe(
                    df
                )

                if prepared is None:

                    debug_rows.append(
                        {
                            "Paese": country,
                            "Codice": code,
                            "Campionato": name,
                            "Stato": "FORMATO NON VALIDO",
                            "Partite": 0,
                            "Errore": "Colonne richieste mancanti",
                        }
                    )

                else:

                    debug_rows.append(
                        {
                            "Paese": country,
                            "Codice": code,
                            "Campionato": name,
                            "Stato": "OK",
                            "Partite": len(
                                prepared
                            ),
                            "Errore": "",
                        }
                    )

            progress.progress(
                (index + 1) / total
            )

        debug_df = pd.DataFrame(
            debug_rows
        )

        st.session_state[
            "debug_df"
        ] = debug_df

    if "debug_df" in st.session_state:

        debug_df = st.session_state[
            "debug_df"
        ]

        st.dataframe(
            debug_df,
            use_container_width=True,
            hide_index=True,
        )

        ok_count = int(
            (
                debug_df["Stato"]
                == "OK"
            ).sum()
        )

        error_count = len(
            debug_df
        ) - ok_count

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Campionati controllati",
                len(debug_df),
            )

        with c2:
            st.metric(
                "OK",
                ok_count,
            )

        with c3:
            st.metric(
                "Con problemi",
                error_count,
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Football Analyzer — modello statistico sperimentale. "
    "Le probabilità sono stime matematiche e non costituiscono "
    "garanzia dell'esito di una partita."
)
