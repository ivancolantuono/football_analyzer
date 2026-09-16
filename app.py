import streamlit as st
import pandas as pd
import numpy as np
import math
import io
import requests


# ============================================================
# CONFIGURAZIONE STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Football Analyzer",
    page_icon="⚽",
    layout="wide"
)


# ============================================================
# STILE
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 0;
}

.subtitle {
    font-size: 16px;
    color: #777;
    margin-bottom: 25px;
}

.card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #ddd;
    background: #fafafa;
    margin-bottom: 12px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 25px;
}

.value-positive {
    color: #008000;
    font-weight: 700;
}

.value-negative {
    color: #cc0000;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# URL
# ============================================================

BASE_URL = "https://www.football-data.co.uk/mmz4281"

FIXTURES_URL = (
    "https://football-data.co.uk/"
    "matches/resources/fixtures.csv"
)


# ============================================================
# CAMPIONATI
# ============================================================

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


# ============================================================
# STAGIONE
# ============================================================

def current_season():

    today = pd.Timestamp.today()

    if today.month >= 7:
        y1 = today.year
        y2 = today.year + 1
    else:
        y1 = today.year - 1
        y2 = today.year

    return (
        f"{str(y1)[-2:]}"
        f"{str(y2)[-2:]}"
    )


# ============================================================
# DOWNLOAD CAMPIONATO
# ============================================================

@st.cache_data(ttl=3600)
def load_league(
    league_code,
    season
):

    url = (
        f"{BASE_URL}/"
        f"{season}/"
        f"{league_code}.csv"
    )

    try:

        response = requests.get(
            url,
            timeout=25,
            headers={
                "User-Agent":
                    "Mozilla/5.0"
            }
        )

        if response.status_code != 200:
            return None

        if len(response.content) < 500:
            return None

        df = pd.read_csv(
            io.BytesIO(
                response.content
            ),
            encoding="latin1"
        )

        return df

    except Exception:

        return None


# ============================================================
# DOWNLOAD FIXTURE FUTURE
# ============================================================

@st.cache_data(ttl=900)
def load_fixtures():

    try:

        response = requests.get(
            FIXTURES_URL,
            timeout=30,
            headers={
                "User-Agent":
                    "Mozilla/5.0"
            }
        )

        if response.status_code != 200:
            return None

        df = pd.read_csv(
            io.BytesIO(
                response.content
            ),
            encoding="latin1"
        )

        df.columns = [
            str(c).strip()
            for c in df.columns
        ]

        return df

    except Exception:

        return None


# ============================================================
# PREPARAZIONE DATAFRAME
# ============================================================

def prepare_dataframe(df):

    if df is None:
        return None

    if df.empty:
        return None

    df = df.copy()

    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce",
            dayfirst=True
        )

    numeric_columns = [

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

        "AvgH",
        "AvgD",
        "AvgA",

        "Avg>2.5",
        "Avg<2.5",

        "PSH",
        "PSD",
        "PSA"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


# ============================================================
# POISSON
# ============================================================

def poisson_probability(
    k,
    lam
):

    if lam <= 0:
        return 0.0

    return (
        math.exp(-lam)
        *
        lam ** k
        /
        math.factorial(k)
    )


# ============================================================
# MATRICE RISULTATI
# ============================================================

def score_matrix(
    home_xg,
    away_xg,
    max_goals=8
):

    matrix = np.zeros(
        (
            max_goals + 1,
            max_goals + 1
        )
    )

    for h in range(
        max_goals + 1
    ):

        for a in range(
            max_goals + 1
        ):

            matrix[h, a] = (

                poisson_probability(
                    h,
                    home_xg
                )

                *

                poisson_probability(
                    a,
                    away_xg
                )
            )

    total = matrix.sum()

    if total > 0:
        matrix /= total

    return matrix


# ============================================================
# PROBABILITÀ
# ============================================================

def calculate_probabilities(
    home_xg,
    away_xg
):

    matrix = score_matrix(
        home_xg,
        away_xg
    )

    home_win = np.tril(
        matrix,
        -1
    ).sum()

    draw = np.trace(
        matrix
    )

    away_win = np.triu(
        matrix,
        1
    ).sum()

    probabilities = {

        "1": home_win,
        "X": draw,
        "2": away_win
    }

    for line in [
        0.5,
        1.5,
        2.5,
        3.5,
        4.5
    ]:

        over = 0.0

        for h in range(
            matrix.shape[0]
        ):

            for a in range(
                matrix.shape[1]
            ):

                if h + a > line:

                    over += matrix[h, a]

        probabilities[
            f"Over {line}"
        ] = over

        probabilities[
            f"Under {line}"
        ] = 1 - over

    # BTTS

    btts = matrix[1:, 1:].sum()

    probabilities["Goal"] = btts

    probabilities["No Goal"] = (
        1 - btts
    )

    # Doppia chance

    probabilities["1X"] = (
        home_win + draw
    )

    probabilities["X2"] = (
        draw + away_win
    )

    probabilities["12"] = (
        home_win + away_win
    )

    # Casa / ospite segna

    probabilities["Casa segna"] = (
        1 - matrix[0, :].sum()
    )

    probabilities["Ospite segna"] = (
        1 - matrix[:, 0].sum()
    )

    # Clean sheet

    probabilities["Casa clean sheet"] = (
        matrix[:, 0].sum()
    )

    probabilities["Ospite clean sheet"] = (
        matrix[0, :].sum()
    )

    return probabilities, matrix


# ============================================================
# FORZA SQUADRE
# ============================================================

def calculate_strengths(df):

    played = df.dropna(
        subset=[
            "FTHG",
            "FTAG"
        ]
    ).copy()

    if played.empty:
        return None

    if (
        "HomeTeam" not in played.columns
        or
        "AwayTeam" not in played.columns
    ):
        return None

    league_home_avg = (
        played["FTHG"].mean()
    )

    league_away_avg = (
        played["FTAG"].mean()
    )

    teams = set(
        played["HomeTeam"].dropna()
    ).union(
        set(
            played["AwayTeam"].dropna()
        )
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

        home_scored = home[
            "FTHG"
        ].sum()

        home_conceded = home[
            "FTAG"
        ].sum()

        away_scored = away[
            "FTAG"
        ].sum()

        away_conceded = away[
            "FTHG"
        ].sum()

        stats[team] = {

            "games":
                home_games + away_games,

            "home_scored":
                (
                    home_scored
                    /
                    home_games
                    if home_games > 0
                    else league_home_avg
                ),

            "home_conceded":
                (
                    home_conceded
                    /
                    home_games
                    if home_games > 0
                    else league_away_avg
                ),

            "away_scored":
                (
                    away_scored
                    /
                    away_games
                    if away_games > 0
                    else league_away_avg
                ),

            "away_conceded":
                (
                    away_conceded
                    /
                    away_games
                    if away_games > 0
                    else league_home_avg
                )
        }

    return (
        stats,
        league_home_avg,
        league_away_avg
    )


# ============================================================
# XG
# ============================================================

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

    home = stats[
        home_team
    ]

    away = stats[
        away_team
    ]

    if home["games"] < 2:
        return None, None

    if away["games"] < 2:
        return None, None

    home_attack = (
        home["home_scored"]
        /
        league_home_avg
        if league_home_avg > 0
        else 1
    )

    home_defense = (
        home["home_conceded"]
        /
        league_away_avg
        if league_away_avg > 0
        else 1
    )

    away_attack = (
        away["away_scored"]
        /
        league_away_avg
        if league_away_avg > 0
        else 1
    )

    away_defense = (
        away["away_conceded"]
        /
        league_home_avg
        if league_home_avg > 0
        else 1
    )

    home_xg = (
        league_home_avg
        *
        home_attack
        *
        away_defense
    )

    away_xg = (
        league_away_avg
        *
        away_attack
        *
        home_defense
    )

    # Limiti di sicurezza

    home_xg = max(
        0.05,
        min(
            home_xg,
            5.0
        )
    )

    away_xg = max(
        0.05,
        min(
            away_xg,
            5.0
        )
    )

    return (
        home_xg,
        away_xg
    )


# ============================================================
# MATCH SQUADRE
# ============================================================

def normalize_team_name(
    name
):

    if pd.isna(name):
        return ""

    name = str(name)

    replacements = {

        "Man United":
            "Man Utd",

        "Manchester United":
            "Man Utd",

        "Man City":
            "Man City",

        "Manchester City":
            "Man City",

        "Nott'm Forest":
            "Nott'm Forest",

        "Nottm Forest":
            "Nott'm Forest",

        "Wolves":
            "Wolves",

        "Tottenham":
            "Tottenham",

        "Spurs":
            "Tottenham",

        "Newcastle":
            "Newcastle",

        "West Ham":
            "West Ham",

        "Leicester":
            "Leicester",

        "Leeds":
            "Leeds"
    }

    name = name.strip()

    return replacements.get(
        name,
        name
    )


# ============================================================
# QUOTE FIXTURE
# ============================================================

def extract_fixture_odds(
    row
):

    odds = {}

    mappings = {

        "1": [
            "MaxH",
            "AvgH",
            "B365H",
            "PSH"
        ],

        "X": [
            "MaxD",
            "AvgD",
            "B365D",
            "PSD"
        ],

        "2": [
            "MaxA",
            "AvgA",
            "B365A",
            "PSA"
        ],

        "Over 2.5": [
            "Max>2.5",
            "Avg>2.5",
            "B365>2.5"
        ],

        "Under 2.5": [
            "Max<2.5",
            "Avg<2.5",
            "B365<2.5"
        ]
    }

    for market, columns in mappings.items():

        values = []

        for col in columns:

            if col not in row.index:
                continue

            try:

                value = float(
                    row[col]
                )

                if (
                    np.isfinite(value)
                    and
                    value > 1
                ):

                    values.append(
                        value
                    )

            except Exception:
                continue

        if values:

            odds[market] = max(
                values
            )

    return odds


# ============================================================
# DEBUG SCANNER
# ============================================================

def run_debug():

    st.subheader(
        "🛠️ Controllo dati"
    )

    fixtures = load_fixtures()

    if fixtures is None:

        st.error(
            "❌ Impossibile scaricare fixtures.csv"
        )

        return

    st.success(
        f"✅ Fixtures scaricate: "
        f"{len(fixtures)} righe"
    )

    st.write(
        "Colonne trovate:"
    )

    st.code(
        ", ".join(
            fixtures.columns.astype(str)
        )
    )

    if "Date" in fixtures.columns:

        fixtures["Date"] = pd.to_datetime(
            fixtures["Date"],
            errors="coerce",
            dayfirst=True
        )

    today = pd.Timestamp.today()

    if "Date" in fixtures.columns:

        future = fixtures[
            fixtures["Date"] >= today
        ]

        st.write(
            f"Partite future: "
            f"**{len(future)}**"
        )

    # Test Italia

    st.markdown(
        "### 🇮🇹 Test Serie A"
    )

    df = load_league(
        "I1",
        current_season()
    )

    if df is None:

        st.error(
            "CSV Serie A non disponibile"
        )

        return

    df = prepare_dataframe(df)

    st.write(
        f"Righe Serie A: "
        f"**{len(df)}**"
    )

    if "FTHG" in df.columns:

        played = df[
            df["FTHG"].notna()
        ]

        future_results = df[
            df["FTHG"].isna()
        ]

        st.write(
            f"Partite giocate: "
            f"**{len(played)}**"
        )

        st.write(
            f"Partite future nel CSV: "
            f"**{len(future_results)}**"
        )

    # Quote fixture

    st.markdown(
        "### 💰 Quote presenti"
    )

    quote_columns = [
        c for c in [
            "MaxH",
            "MaxD",
            "MaxA",
            "AvgH",
            "AvgD",
            "AvgA",
            "Max>2.5",
            "Max<2.5",
            "Avg>2.5",
            "Avg<2.5"
        ]
        if c in fixtures.columns
    ]

    if quote_columns:

        st.write(
            quote_columns
        )

        st.dataframe(
            fixtures[
                quote_columns
            ].head(10),
            use_container_width=True
        )

    else:

        st.warning(
            "⚠️ Nessuna colonna quote "
            "riconosciuta."
        )


# ============================================================
# VALUE SCANNER
# ============================================================

def scan_all_leagues(
    min_probability=0.50,
    min_value=0.00
):

    fixtures = load_fixtures()

    if fixtures is None:

        return (
            pd.DataFrame(),
            {
                "fixtures": 0,
                "future": 0,
                "leagues": 0,
                "matched": 0,
                "with_odds": 0,
                "signals": 0
            }
        )

    fixtures = fixtures.copy()

    fixtures.columns = [
        str(c).strip()
        for c in fixtures.columns
    ]

    if "Date" in fixtures.columns:

        fixtures["Date"] = pd.to_datetime(
            fixtures["Date"],
            errors="coerce",
            dayfirst=True
        )

    # --------------------------------------------------------
    # FUTURE
    # --------------------------------------------------------

    today = pd.Timestamp.today().normalize()

    if "Date" in fixtures.columns:

        future = fixtures[
            fixtures["Date"] >= today
        ].copy()

    else:

        future = fixtures.copy()

    results = []

    debug = {

        "fixtures":
            len(fixtures),

        "future":
            len(future),

        "leagues":
            0,

        "matched":
            0,

        "with_odds":
            0,

        "signals":
            0
    }

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    progress = st.progress(0)

    total = len(
        LEAGUES
    )

    # --------------------------------------------------------
    # CAMPIONATI
    # --------------------------------------------------------

    for idx, (
        league_name,
        league_code
    ) in enumerate(
        LEAGUES.items()
    ):

        progress.progress(
            int(
                (
                    idx + 1
                )
                /
                total
                *
                100
            )
        )

        df = load_league(
            league_code,
            current_season()
        )

        if df is None:
            continue

        df = prepare_dataframe(
            df
        )

        if df is None:
            continue

        strength = calculate_strengths(
            df
        )

        if strength is None:
            continue

        (
            stats,
            league_home_avg,
            league_away_avg
        ) = strength

        debug[
            "leagues"
        ] += 1

        # ----------------------------------------------------
        # FIXTURE
        # ----------------------------------------------------

        for _, fixture in future.iterrows():

            home_raw = fixture.get(
                "HomeTeam"
            )

            away_raw = fixture.get(
                "AwayTeam"
            )

            if pd.isna(
                home_raw
            ):
                continue

            if pd.isna(
                away_raw
            ):
                continue

            home = normalize_team_name(
                home_raw
            )

            away = normalize_team_name(
                away_raw
            )

            # ------------------------------------------------
            # Match squadra
            # ------------------------------------------------

            if home not in stats:
                continue

            if away not in stats:
                continue

            debug[
                "matched"
            ] += 1

            # ------------------------------------------------
            # XG
            # ------------------------------------------------

            home_xg, away_xg = calculate_xg(
                home,
                away,
                stats,
                league_home_avg,
                league_away_avg
            )

            if home_xg is None:
                continue

            # ------------------------------------------------
            # PROBABILITÀ
            # ------------------------------------------------

            probabilities, _ = (
                calculate_probabilities(
                    home_xg,
                    away_xg
                )
            )

            # ------------------------------------------------
            # QUOTE
            # ------------------------------------------------

            odds = extract_fixture_odds(
                fixture
            )

            if not odds:
                continue

            debug[
                "with_odds"
            ] += 1

            # ------------------------------------------------
            # MERCATI
            # ------------------------------------------------

            markets = {

                "1":
                    probabilities["1"],

                "X":
                    probabilities["X"],

                "2":
                    probabilities["2"],

                "Over 2.5":
                    probabilities[
                        "Over 2.5"
                    ],

                "Under 2.5":
                    probabilities[
                        "Under 2.5"
                    ]
            }

            # ------------------------------------------------
            # VALUE
            # ------------------------------------------------

            for market, probability in (
                markets.items()
            ):

                if market not in odds:
                    continue

                odd = odds[
                    market
                ]

                if probability <= 0:
                    continue

                fair_odd = (
                    1
                    /
                    probability
                )

                value = (
                    probability
                    *
                    odd
                ) - 1

                if (
                    probability
                    <
                    min_probability
                ):
                    continue

                if (
                    value
                    <
                    min_value
                ):
                    continue

                home_games = stats[
                    home
                ]["games"]

                away_games = stats[
                    away
                ]["games"]

                reliability = min(
                    100,
                    (
                        home_games
                        +
                        away_games
                    )
                    /
                    20
                    *
                    100
                )

                match_date = ""

                if (
                    "Date"
                    in fixture.index
                    and
                    not pd.isna(
                        fixture["Date"]
                    )
                ):

                    match_date = (
                        fixture["Date"]
                        .strftime(
                            "%d/%m/%Y"
                        )
                    )

                results.append({

                    "Campionato":
                        league_name,

                    "Data":
                        match_date,

                    "Partita":
                        f"{home} - {away}",

                    "Mercato":
                        market,

                    "Probabilità":
                        probability,

                    "Quota":
                        odd,

                    "Quota equa":
                        fair_odd,

                    "Value":
                        value,

                    "xG casa":
                        home_xg,

                    "xG ospite":
                        away_xg,

                    "Affidabilità":
                        reliability
                })

                debug[
                    "signals"
                ] += 1

    progress.empty()

    if not results:

        return (
            pd.DataFrame(),
            debug
        )

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        [
            "Value",
            "Probabilità"
        ],
        ascending=False
    )

    return (
        result_df.reset_index(
            drop=True
        ),
        debug
    )


# ============================================================
# STATISTICHE SQUADRA
# ============================================================

def team_stats(
    df,
    team
):

    matches = df[
        (
            df["HomeTeam"]
            == team
        )
        |
        (
            df["AwayTeam"]
            == team
        )
    ].dropna(
        subset=[
            "FTHG",
            "FTAG"
        ]
    )

    if matches.empty:
        return None

    gf = []
    ga = []

    wins = 0
    draws = 0
    losses = 0

    over25 = 0
    btts = 0

    for _, row in matches.iterrows():

        if row["HomeTeam"] == team:

            goals_for = row["FTHG"]
            goals_against = row["FTAG"]

        else:

            goals_for = row["FTAG"]
            goals_against = row["FTHG"]

        gf.append(
            goals_for
        )

        ga.append(
            goals_against
        )

        if goals_for > goals_against:
            wins += 1

        elif goals_for == goals_against:
            draws += 1

        else:
            losses += 1

        if (
            goals_for
            +
            goals_against
            >
            2
        ):
            over25 += 1

        if (
            goals_for > 0
            and
            goals_against > 0
        ):
            btts += 1

    n = len(
        matches
    )

    return {

        "Partite":
            n,

        "Gol fatti":
            np.mean(gf),

        "Gol subiti":
            np.mean(ga),

        "Vittorie":
            wins,

        "Pareggi":
            draws,

        "Sconfitte":
            losses,

        "Over 2.5":
            over25 / n,

        "Goal":
            btts / n
    }


# ============================================================
# ANALISI PARTITA
# ============================================================

def analyze_match(
    df,
    home,
    away
):

    strength = calculate_strengths(
        df
    )

    if strength is None:
        return None

    (
        stats,
        league_home_avg,
        league_away_avg
    ) = strength

    home_xg, away_xg = calculate_xg(
        home,
        away,
        stats,
        league_home_avg,
        league_away_avg
    )

    if home_xg is None:
        return None

    probabilities, matrix = (
        calculate_probabilities(
            home_xg,
            away_xg
        )
    )

    return (
        home_xg,
        away_xg,
        probabilities,
        matrix
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '⚽ Football Analyzer'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Probabilità • xG • Poisson • Value Scanner'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚽ Football Analyzer"
)

page = st.sidebar.radio(
    "Sezione",
    [
        "🏠 Dashboard",
        "📊 Analisi partita",
        "🔎 Value Scanner",
        "🛠️ Debug dati"
    ]
)

st.sidebar.markdown(
    "---"
)

st.sidebar.write(
    f"Stagione: **{current_season()}**"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header(
        "🏠 Dashboard"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Campionati",
            len(LEAGUES)
        )

    with c2:

        st.metric(
            "Modello",
            "Poisson"
        )

    with c3:

        st.metric(
            "Analisi",
            "xG + Value"
        )

    st.markdown(
        "---"
    )

    st.write(
        """
        Il programma utilizza i dati storici delle squadre
        per calcolare forza offensiva e difensiva.

        Da questi dati vengono stimati i gol attesi (xG).

        La distribuzione di Poisson viene poi utilizzata
        per ottenere le probabilità dei risultati.

        Il Value Scanner confronta queste probabilità
        con le quote disponibili sulle fixture future.
        """
    )

    st.info(
        "Il modello produce stime statistiche. "
        "Non esiste garanzia sull'esito di una partita."
    )


# ============================================================
# ANALISI PARTITA
# ============================================================

elif page == "📊 Analisi partita":

    st.header(
        "📊 Analisi partita"
    )

    league = st.selectbox(
        "Campionato",
        list(LEAGUES.keys())
    )

    df = load_league(
        LEAGUES[league],
        current_season()
    )

    if df is None:

        st.error(
            "Dati non disponibili."
        )

        st.stop()

    df = prepare_dataframe(
        df
    )

    teams = sorted(
        list(
            set(
                df["HomeTeam"].dropna()
            )
            |
            set(
                df["AwayTeam"].dropna()
            )
        )
    )

    c1, c2 = st.columns(2)

    with c1:

        home = st.selectbox(
            "Casa",
            teams
        )

    with c2:

        away_list = [
            x for x in teams
            if x != home
        ]

        away = st.selectbox(
            "Ospite",
            away_list
        )

    if st.button(
        "🔍 ANALIZZA",
        use_container_width=True
    ):

        result = analyze_match(
            df,
            home,
            away
        )

        if result is None:

            st.error(
                "Dati insufficienti."
            )

            st.stop()

        (
            home_xg,
            away_xg,
            probabilities,
            matrix
        ) = result

        # ----------------------------------------------------
        # XG
        # ----------------------------------------------------

        st.subheader(
            "⚽ Gol attesi"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                home,
                f"{home_xg:.2f}"
            )

        with c2:
            st.metric(
                "Totale xG",
                f"{home_xg + away_xg:.2f}"
            )

        with c3:
            st.metric(
                away,
                f"{away_xg:.2f}"
            )

        # ----------------------------------------------------
        # 1X2
        # ----------------------------------------------------

        st.subheader(
            "1X2"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "1",
                f"{probabilities['1'] * 100:.1f}%"
            )

        with c2:

            st.metric(
                "X",
                f"{probabilities['X'] * 100:.1f}%"
            )

        with c3:

            st.metric(
                "2",
                f"{probabilities['2'] * 100:.1f}%"
            )

        # ----------------------------------------------------
        # DOPPIA CHANCE
        # ----------------------------------------------------

        st.subheader(
            "Doppia chance"
        )

        dc = pd.DataFrame({

            "Mercato": [
                "1X",
                "X2",
                "12"
            ],

            "Probabilità": [

                probabilities["1X"],
                probabilities["X2"],
                probabilities["12"]
            ]
        })

        dc["Probabilità"] = (
            dc["Probabilità"]
            * 100
        ).round(1).astype(str) + "%"

        st.dataframe(
            dc,
            hide_index=True,
            use_container_width=True
        )

        # ----------------------------------------------------
        # OVER UNDER
        # ----------------------------------------------------

        st.subheader(
            "⚽ Over / Under"
        )

        rows = []

        for line in [
            0.5,
            1.5,
            2.5,
            3.5,
            4.5
        ]:

            rows.append({

                "Linea":
                    line,

                "Over":
                    probabilities[
                        f"Over {line}"
                    ],

                "Under":
                    probabilities[
                        f"Under {line}"
                    ]
            })

        ou = pd.DataFrame(
            rows
        )

        ou["Over"] = (
            ou["Over"]
            * 100
        ).round(1).astype(str) + "%"

        ou["Under"] = (
            ou["Under"]
            * 100
        ).round(1).astype(str) + "%"

        st.dataframe(
            ou,
            hide_index=True,
            use_container_width=True
        )

        # ----------------------------------------------------
        # GOAL
        # ----------------------------------------------------

        st.subheader(
            "🎯 Goal / No Goal"
        )

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Goal",
                f"{probabilities['Goal'] * 100:.1f}%"
            )

        with c2:

            st.metric(
                "No Goal",
                f"{probabilities['No Goal'] * 100:.1f}%"
            )

        # ----------------------------------------------------
        # RISULTATI ESATTI
        # ----------------------------------------------------

        st.subheader(
            "🎯 Risultati esatti"
        )

        scores = []

        for h in range(
            matrix.shape[0]
        ):

            for a in range(
                matrix.shape[1]
            ):

                scores.append({

                    "Risultato":
                        f"{h}-{a}",

                    "Probabilità":
                        matrix[h, a]
                })

        scores = pd.DataFrame(
            scores
        )

        scores = scores.sort_values(
            "Probabilità",
            ascending=False
        ).head(10)

        scores["Probabilità"] = (
            scores["Probabilità"]
            * 100
        ).round(2).astype(str) + "%"

        st.dataframe(
            scores,
            hide_index=True,
            use_container_width=True
        )

        # ----------------------------------------------------
        # STATISTICHE
        # ----------------------------------------------------

        st.subheader(
            "📈 Statistiche"
        )

        home_stats = team_stats(
            df,
            home
        )

        away_stats = team_stats(
            df,
            away
        )

        if (
            home_stats
            and
            away_stats
        ):

            stats_df = pd.DataFrame({

                home:
                    home_stats,

                away:
                    away_stats
            })

            st.dataframe(
                stats_df,
                use_container_width=True
            )


# ============================================================
# VALUE SCANNER
# ============================================================

elif page == "🔎 Value Scanner":

    st.header(
        "🔎 VALUE SCANNER"
    )

    st.write(
        """
        Analizzo le fixture future disponibili,
        calcolo le probabilità del modello e confronto
        la probabilità con la quota.
        """
    )

    # --------------------------------------------------------
    # FILTRI
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        min_probability = st.slider(
            "Probabilità minima",
            0.40,
            0.90,
            0.50,
            0.01
        )

    with c2:

        min_value = st.slider(
            "Value minimo",
            -0.10,
            0.30,
            0.00,
            0.01
        )

    with c3:

        max_results = st.number_input(
            "Risultati da mostrare",
            5,
            100,
            50,
            5
        )

    st.markdown(
        "---"
    )

    if st.button(
        "🚀 SCANSIONA TUTTI I CAMPIONATI",
        use_container_width=True
    ):

        with st.spinner(
            "Scarico fixture e analizzo i campionati..."
        ):

            results, debug = (
                scan_all_leagues(
                    min_probability,
                    min_value
                )
            )

        # ----------------------------------------------------
        # DEBUG RIASSUNTIVO
        # ----------------------------------------------------

        st.subheader(
            "🔎 Controllo scansione"
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.metric(
                "Fixture",
                debug["fixtures"]
            )

        with c2:
            st.metric(
                "Future",
                debug["future"]
            )

        with c3:
            st.metric(
                "Campionati",
                debug["leagues"]
            )

        with c4:
            st.metric(
                "Match abbinati",
                debug["matched"]
            )

        with c5:
            st.metric(
                "Con quote",
                debug["with_odds"]
            )

        # ----------------------------------------------------
        # NESSUN RISULTATO
        # ----------------------------------------------------

        if results.empty:

            st.warning(
                "⚠️ Nessun segnale trovato."
            )

            st.write(
                f"""
                Fixture scaricate: **{debug['fixtures']}**

                Fixture future: **{debug['future']}**

                Campionati caricati:
                **{debug['leagues']}**

                Partite abbinate:
                **{debug['matched']}**

                Partite con quote:
                **{debug['with_odds']}**

                Segnali:
                **{debug['signals']}**
                """
            )

            st.info(
                "Prova temporaneamente "
                "Probabilità minima = 40% "
                "e Value minimo = -10% "
                "per verificare i dati."
            )

        else:

            results = results.head(
                int(max_results)
            )

            # ------------------------------------------------
            # KPI
            # ------------------------------------------------

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Segnali",
                    len(results)
                )

            with c2:

                st.metric(
                    "Value medio",
                    f"{results['Value'].mean() * 100:.1f}%"
                )

            with c3:

                st.metric(
                    "Probabilità media",
                    f"{results['Probabilità'].mean() * 100:.1f}%"
                )

            with c4:

                st.metric(
                    "Quota media",
                    f"{results['Quota'].mean():.2f}"
                )

            st.markdown(
                "---"
            )

            # ------------------------------------------------
            # TABELLA
            # ------------------------------------------------

            display = results.copy()

            display["Probabilità"] = (
                display["Probabilità"]
                * 100
            ).round(1).astype(str) + "%"

            display["Value"] = (
                display["Value"]
                * 100
            ).round(1).astype(str) + "%"

            display["Affidabilità"] = (
                display["Affidabilità"]
            ).round(0).astype(int).astype(str) + "%"

            display["Quota"] = (
                display["Quota"]
                .round(2)
            )

            display["Quota equa"] = (
                display["Quota equa"]
                .round(2)
            )

            display["xG casa"] = (
                display["xG casa"]
                .round(2)
            )

            display["xG ospite"] = (
                display["xG ospite"]
                .round(2)
            )

            st.dataframe(
                display,
                hide_index=True,
                use_container_width=True
            )

            # ------------------------------------------------
            # TOP SEGNALI
            # ------------------------------------------------

            st.subheader(
                "🏆 Migliori segnali"
            )

            for i, (_, row) in enumerate(
                results.head(10).iterrows(),
                start=1
            ):

                st.markdown(
                    f"""
                    <div class="card">

                    <b>#{i} — {row['Partita']}</b>

                    <br>

                    {row['Campionato']}

                    <br><br>

                    <b>Mercato:</b>
                    {row['Mercato']}

                    &nbsp;&nbsp;

                    <b>Probabilità:</b>
                    {row['Probabilità'] * 100:.1f}%

                    &nbsp;&nbsp;

                    <b>Quota:</b>
                    {row['Quota']:.2f}

                    &nbsp;&nbsp;

                    <b>Value:</b>
                    <span class="value-positive">
                    +{row['Value'] * 100:.1f}%
                    </span>

                    <br>

                    xG:
                    {row['xG casa']:.2f}
                    -
                    {row['xG ospite']:.2f}

                    &nbsp;&nbsp;

                    Affidabilità:
                    {row['Affidabilità']:.0f}%

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# DEBUG
# ============================================================

elif page == "🛠️ Debug dati":

    st.header(
        "🛠️ Debug dati"
    )

    st.write(
        """
        Questa pagina serve per capire esattamente
        quali dati vengono ricevuti da Football-Data.
        """
    )

    if st.button(
        "🔍 CONTROLLA DATI"
    ):

        run_debug()
```

### `requirements.txt`

Usa solo questo:

```txt
streamlit
pandas
numpy
requests
```

### Cosa cambia rispetto a prima

Adesso quando premi **SCANSIONA TUTTI I CAMPIONATI**, non vedrai più semplicemente "nessun risultato".

Vedrai qualcosa del genere:

```text
st.subheader("🔎 CONTROLLO SCANSIONE")

Fixture             250
Future               82
Campionati           30
Match abbinati       64
Con quote            58
Segnali              12
```

Questa informazione è fondamentale.

Se invece vediamo:

```text
Fixture             250
Future               82
Campionati           30
Match abbinati        0
Con quote             0
Segnali               0
```

allora sappiamo che il problema è **l'abbinamento dei nomi delle squadre**, non il Value.

Se vediamo:

```text
Fixture             250
Future               82
Campionati           30
Match abbinati       64
Con quote             0
Segnali               0
```

allora il problema sono **le colonne delle quote**.

Se invece:

```text
Fixture             250
Future               82
Campionati           30
Match abbinati       64
Con quote            58
Segnali               0
```

allora il motore funziona, ma **nessuna partita supera i filtri**.

Questo è molto meglio perché adesso possiamo correggere il punto preciso senza andare a tentativi.

Football-Data conferma inoltre che le fixture future vengono pubblicate con quote 1X2, total goals e Asian handicap e che il file viene aggiornato per gli incontri del weekend e per quelli infrasettimanali.

**Una nota importante:** la lista che abbiamo messo contiene i campionati che Football-Data rende disponibili con questi codici; non sono letteralmente "tutti i campionati del mondo". Il sito indica anche 16 divisioni extra (tra cui Austria, Cina, Danimarca, Norvegia, Svezia, Svizzera ecc.), che possiamo aggiungere subito dopo aver verificato il funzionamento dello scanner.

Dopo aver caricato questo `app.py`, **vai direttamente su `🛠️ Debug dati` e premi `🔍 CONTROLLA DATI`**. Poi su `🔎 Value Scanner` premi la scansione. Con quei numeri possiamo sistemare definitivamente l'eventuale collo di bottiglia.
