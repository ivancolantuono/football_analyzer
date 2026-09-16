import streamlit as st
import pandas as pd
import numpy as np
import math
import io
import requests


# ============================================================
# CONFIGURAZIONE
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
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    background: #fafafa;
    margin-bottom: 15px;
}

.big-number {
    font-size: 30px;
    font-weight: 700;
}

.small-label {
    color: #777;
    font-size: 14px;
}

.value-positive {
    font-weight: 700;
    color: #008000;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


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


BASE_URL = "https://www.football-data.co.uk/mmz4281"


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

    return f"{str(y1)[-2:]}{str(y2)[-2:]}"


# ============================================================
# DOWNLOAD CAMPIONATO
# ============================================================

@st.cache_data(ttl=3600)
def load_league(league_code, season):

    url = f"{BASE_URL}/{season}/{league_code}.csv"

    try:

        response = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        if response.status_code != 200:
            return None

        if len(response.content) < 500:
            return None

        df = pd.read_csv(
            io.BytesIO(response.content),
            encoding="latin1"
        )

        return df

    except Exception:

        return None


# ============================================================
# PREPARAZIONE DATI
# ============================================================

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

def poisson_probability(k, lam):

    if lam <= 0:
        return 0

    return (
        math.exp(-lam)
        * lam ** k
        / math.factorial(k)
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
        (max_goals + 1, max_goals + 1)
    )

    for home_goals in range(max_goals + 1):

        for away_goals in range(max_goals + 1):

            matrix[
                home_goals,
                away_goals
            ] = (

                poisson_probability(
                    home_goals,
                    home_xg
                )

                *

                poisson_probability(
                    away_goals,
                    away_xg
                )
            )

    total = matrix.sum()

    if total > 0:
        matrix = matrix / total

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

    draw = np.trace(matrix)

    away_win = np.triu(
        matrix,
        1
    ).sum()

    probabilities = {

        "1": home_win,

        "X": draw,

        "2": away_win,

    }

    # OVER / UNDER

    for line in [
        0.5,
        1.5,
        2.5,
        3.5,
        4.5
    ]:

        over = 0

        for h in range(matrix.shape[0]):

            for a in range(matrix.shape[1]):

                if h + a > line:

                    over += matrix[h, a]

        probabilities[
            f"Over {line}"
        ] = over

        probabilities[
            f"Under {line}"
        ] = 1 - over

    # BTTS

    btts = 0

    for h in range(1, matrix.shape[0]):

        for a in range(1, matrix.shape[1]):

            btts += matrix[h, a]

    probabilities["Goal"] = btts
    probabilities["No Goal"] = 1 - btts

    # DOPPIA CHANCE

    probabilities["1X"] = home_win + draw
    probabilities["X2"] = draw + away_win
    probabilities["12"] = home_win + away_win

    # SQUADRA CASA SEGNA

    home_score = 1 - matrix[0, :].sum()

    away_score = 1 - matrix[:, 0].sum()

    probabilities["Casa segna"] = home_score
    probabilities["Ospite segna"] = away_score

    probabilities["Casa clean sheet"] = matrix[:, 0].sum()
    probabilities["Ospite clean sheet"] = matrix[0, :].sum()

    return probabilities, matrix


# ============================================================
# STATISTICHE SQUADRE
# ============================================================

def calculate_team_stats(
    df,
    team
):

    matches = df[
        (
            df["HomeTeam"] == team
        )
        |
        (
            df["AwayTeam"] == team
        )
    ].copy()

    matches = matches.dropna(
        subset=[
            "FTHG",
            "FTAG"
        ]
    )

    if matches.empty:
        return None

    goals_for = []

    goals_against = []

    wins = 0
    draws = 0
    losses = 0

    over05 = 0
    over15 = 0
    over25 = 0
    over35 = 0

    btts = 0

    for _, row in matches.iterrows():

        if row["HomeTeam"] == team:

            gf = row["FTHG"]
            ga = row["FTAG"]

        else:

            gf = row["FTAG"]
            ga = row["FTHG"]

        goals_for.append(gf)
        goals_against.append(ga)

        if gf > ga:
            wins += 1

        elif gf == ga:
            draws += 1

        else:
            losses += 1

        total = gf + ga

        if total > 0:
            over05 += 1

        if total > 1:
            over15 += 1

        if total > 2:
            over25 += 1

        if total > 3:
            over35 += 1

        if gf > 0 and ga > 0:
            btts += 1

    n = len(matches)

    return {

        "Partite": n,

        "Gol fatti":
            np.mean(goals_for),

        "Gol subiti":
            np.mean(goals_against),

        "Vittorie":
            wins,

        "Pareggi":
            draws,

        "Sconfitte":
            losses,

        "Over 0.5":
            over05 / n,

        "Over 1.5":
            over15 / n,

        "Over 2.5":
            over25 / n,

        "Over 3.5":
            over35 / n,

        "Goal":
            btts / n,

    }


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

    league_home_avg = played[
        "FTHG"
    ].mean()

    league_away_avg = played[
        "FTAG"
    ].mean()

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
                    home_scored / home_games
                    if home_games
                    else league_home_avg
                ),

            "home_conceded":
                (
                    home_conceded / home_games
                    if home_games
                    else league_away_avg
                ),

            "away_scored":
                (
                    away_scored / away_games
                    if away_games
                    else league_away_avg
                ),

            "away_conceded":
                (
                    away_conceded / away_games
                    if away_games
                    else league_home_avg
                ),
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

    home = stats[home_team]
    away = stats[away_team]

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

    home_xg = max(
        0.05,
        min(home_xg, 5)
    )

    away_xg = max(
        0.05,
        min(away_xg, 5)
    )

    return home_xg, away_xg


# ============================================================
# QUOTE
# ============================================================

def get_best_odds(row):

    odds = {}

    mapping = {

        "1": [
            "MaxH",
            "BbAvH",
            "B365H"
        ],

        "X": [
            "MaxD",
            "BbAvD",
            "B365D"
        ],

        "2": [
            "MaxA",
            "BbAvA",
            "B365A"
        ],

        "Over 2.5": [
            "Max>2.5",
            "BbAv>2.5",
            "B365>2.5"
        ],

        "Under 2.5": [
            "Max<2.5",
            "BbAv<2.5",
            "B365<2.5"
        ],
    }

    for market, columns in mapping.items():

        values = []

        for col in columns:

            if col in row.index:

                try:

                    value = float(
                        row[col]
                    )

                    if value > 1:
                        values.append(value)

                except Exception:
                    pass

        if values:
            odds[market] = max(values)

    return odds


# ============================================================
# ANALISI PARTITA
# ============================================================

def analyze_match(
    df,
    home_team,
    away_team
):

    strength_data = calculate_strengths(df)

    if strength_data is None:
        return None

    stats, league_home_avg, league_away_avg = strength_data

    home_xg, away_xg = calculate_xg(
        home_team,
        away_team,
        stats,
        league_home_avg,
        league_away_avg
    )

    if home_xg is None:
        return None

    probabilities, matrix = calculate_probabilities(
        home_xg,
        away_xg
    )

    return {
        "home_xg": home_xg,
        "away_xg": away_xg,
        "probabilities": probabilities,
        "matrix": matrix,
        "stats": stats,
        "league_home_avg": league_home_avg,
        "league_away_avg": league_away_avg
    }


# ============================================================
# VALUE
# ============================================================

def calculate_value(
    probability,
    odd
):

    if probability <= 0:
        return None

    fair_odd = 1 / probability

    value = (
        probability * odd
    ) - 1

    return fair_odd, value


# ============================================================
# VALUE SCANNER
# ============================================================

@st.cache_data(ttl=1800)
def scan_all_leagues(
    min_probability=0.55,
    min_value=0.03
):

    all_results = []

    season = current_season()

    for league_name, league_code in LEAGUES.items():

        df = load_league(
            league_code,
            season
        )

        if df is None:
            continue

        df = prepare_dataframe(df)

        if df is None:
            continue

        strength_data = calculate_strengths(
            df
        )

        if strength_data is None:
            continue

        (
            stats,
            league_home_avg,
            league_away_avg
        ) = strength_data

        # PARTITE NON ANCORA GIOCATE

        upcoming = df[
            (
                df["FTHG"].isna()
            )
            |
            (
                df["FTAG"].isna()
            )
        ].copy()

        if upcoming.empty:
            continue

        for _, row in upcoming.iterrows():

            home_team = row.get(
                "HomeTeam"
            )

            away_team = row.get(
                "AwayTeam"
            )

            if pd.isna(home_team):
                continue

            if pd.isna(away_team):
                continue

            try:

                home_xg, away_xg = calculate_xg(
                    home_team,
                    away_team,
                    stats,
                    league_home_avg,
                    league_away_avg
                )

                if home_xg is None:
                    continue

                probabilities, _ = calculate_probabilities(
                    home_xg,
                    away_xg
                )

                odds = get_best_odds(
                    row
                )

                markets = {

                    "1":
                        probabilities["1"],

                    "X":
                        probabilities["X"],

                    "2":
                        probabilities["2"],

                    "Over 2.5":
                        probabilities["Over 2.5"],

                    "Under 2.5":
                        probabilities["Under 2.5"],
                }

                for market, probability in markets.items():

                    if market not in odds:
                        continue

                    odd = odds[market]

                    if probability < min_probability:
                        continue

                    fair_odd, value = calculate_value(
                        probability,
                        odd
                    )

                    if value < min_value:
                        continue

                    games_home = stats[
                        home_team
                    ]["games"]

                    games_away = stats[
                        away_team
                    ]["games"]

                    reliability = min(
                        100,
                        (
                            games_home
                            +
                            games_away
                        )
                        / 20
                        * 100
                    )

                    all_results.append({

                        "Campionato":
                            league_name,

                        "Data":
                            row["Date"].strftime(
                                "%d/%m/%Y"
                            )
                            if not pd.isna(row["Date"])
                            else "",

                        "Partita":
                            f"{home_team} - {away_team}",

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

            except Exception:
                continue

    if not all_results:
        return pd.DataFrame()

    result = pd.DataFrame(
        all_results
    )

    result = result.sort_values(
        "Value",
        ascending=False
    )

    return result.reset_index(
        drop=True
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚽ Football Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analisi statistica, probabilità, xG e Value Scanner'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚽ Football Analyzer")

page = st.sidebar.radio(
    "Sezione",
    [
        "🏠 Dashboard",
        "📊 Analisi partita",
        "🔎 Value Scanner"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    f"Stagione dati: {current_season()}"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header("🏠 Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="card">
                <div class="small-label">
                Campionati
                </div>
                <div class="big-number">
                30+
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="card">
                <div class="small-label">
                Modello
                </div>
                <div class="big-number">
                Poisson
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="card">
                <div class="small-label">
                Analisi
                </div>
                <div class="big-number">
                xG + Value
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.subheader(
        "Come funziona il modello"
    )

    st.write(
        """
        Il programma utilizza i risultati storici delle squadre
        per stimare la forza offensiva e difensiva.

        Da queste informazioni calcola i gol attesi (xG) e utilizza
        una distribuzione di Poisson per stimare la probabilità
        dei diversi risultati.

        Il Value Scanner confronta poi la probabilità del modello
        con la quota disponibile nei dati.
        """
    )

    st.info(
        "Il Value è un indicatore matematico del modello, "
        "non una garanzia dell'esito della partita."
    )


# ============================================================
# ANALISI PARTITA
# ============================================================

elif page == "📊 Analisi partita":

    st.header("📊 Analisi partita")

    league_name = st.selectbox(
        "Campionato",
        list(LEAGUES.keys())
    )

    league_code = LEAGUES[
        league_name
    ]

    with st.spinner(
        "Caricamento dati..."
    ):

        df = load_league(
            league_code,
            current_season()
        )

    if df is None:

        st.error(
            "Dati del campionato non disponibili."
        )

        st.stop()

    df = prepare_dataframe(df)

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

    col1, col2 = st.columns(2)

    with col1:

        home_team = st.selectbox(
            "Squadra di casa",
            teams
        )

    with col2:

        away_options = [
            x for x in teams
            if x != home_team
        ]

        away_team = st.selectbox(
            "Squadra ospite",
            away_options
        )

    if st.button(
        "🔍 ANALIZZA PARTITA",
        use_container_width=True
    ):

        analysis = analyze_match(
            df,
            home_team,
            away_team
        )

        if analysis is None:

            st.error(
                "Impossibile calcolare l'analisi."
            )

            st.stop()

        home_xg = analysis[
            "home_xg"
        ]

        away_xg = analysis[
            "away_xg"
        ]

        probabilities = analysis[
            "probabilities"
        ]

        matrix = analysis[
            "matrix"
        ]

        # ----------------------------------------------------
        # XG
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">⚽ Gol attesi</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                home_team,
                f"{home_xg:.2f} xG"
            )

        with c2:

            st.metric(
                "Totale",
                f"{home_xg + away_xg:.2f}"
            )

        with c3:

            st.metric(
                away_team,
                f"{away_xg:.2f} xG"
            )

        # ----------------------------------------------------
        # 1X2
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">1X2</div>',
            unsafe_allow_html=True
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
            dc["Probabilità"] * 100
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

        ou_rows = []

        for line in [
            0.5,
            1.5,
            2.5,
            3.5,
            4.5
        ]:

            ou_rows.append({

                "Linea": line,

                "Over":
                    probabilities[
                        f"Over {line}"
                    ],

                "Under":
                    probabilities[
                        f"Under {line}"
                    ]
            })

        ou_df = pd.DataFrame(
            ou_rows
        )

        ou_df["Over"] = (
            ou_df["Over"] * 100
        ).round(1).astype(str) + "%"

        ou_df["Under"] = (
            ou_df["Under"] * 100
        ).round(1).astype(str) + "%"

        st.dataframe(
            ou_df,
            hide_index=True,
            use_container_width=True
        )

        # ----------------------------------------------------
        # GOAL / NO GOAL
        # ----------------------------------------------------

        st.subheader(
            "⚽ Goal / No Goal"
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
        # SQUADRE SEGNANO
        # ----------------------------------------------------

        st.subheader(
            "🎯 Squadre"
        )

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                f"{home_team} segna",
                f"{probabilities['Casa segna'] * 100:.1f}%"
            )

            st.metric(
                "Clean sheet casa",
                f"{probabilities['Casa clean sheet'] * 100:.1f}%"
            )

        with c2:

            st.metric(
                f"{away_team} segna",
                f"{probabilities['Ospite segna'] * 100:.1f}%"
            )

            st.metric(
                "Clean sheet ospite",
                f"{probabilities['Ospite clean sheet'] * 100:.1f}%"
            )

        # ----------------------------------------------------
        # RISULTATI ESATTI
        # ----------------------------------------------------

        st.subheader(
            "🎯 Risultati esatti più probabili"
        )

        score_results = []

        for h in range(
            matrix.shape[0]
        ):

            for a in range(
                matrix.shape[1]
            ):

                score_results.append({

                    "Risultato":
                        f"{h}-{a}",

                    "Probabilità":
                        matrix[h, a]
                })

        score_df = pd.DataFrame(
            score_results
        )

        score_df = score_df.sort_values(
            "Probabilità",
            ascending=False
        ).head(10)

        score_df["Probabilità"] = (
            score_df["Probabilità"]
            * 100
        ).round(2).astype(str) + "%"

        st.dataframe(
            score_df,
            hide_index=True,
            use_container_width=True
        )

        # ----------------------------------------------------
        # STATISTICHE
        # ----------------------------------------------------

        st.subheader(
            "📈 Statistiche squadre"
        )

        home_stats = calculate_team_stats(
            df,
            home_team
        )

        away_stats = calculate_team_stats(
            df,
            away_team
        )

        if home_stats and away_stats:

            stat_df = pd.DataFrame({

                home_team:
                    home_stats,

                away_team:
                    away_stats
            })

            st.dataframe(
                stat_df,
                use_container_width=True
            )

        # ----------------------------------------------------
        # ULTIME PARTITE
        # ----------------------------------------------------

        st.subheader(
            "📅 Ultime partite"
        )

        played = df.dropna(
            subset=[
                "FTHG",
                "FTAG"
            ]
        )

        recent_home = played[
            (
                played["HomeTeam"]
                == home_team
            )
            |
            (
                played["AwayTeam"]
                == home_team
            )
        ].tail(5)

        recent_away = played[
            (
                played["HomeTeam"]
                == away_team
            )
            |
            (
                played["AwayTeam"]
                == away_team
            )
        ].tail(5)

        c1, c2 = st.columns(2)

        with c1:

            st.write(
                f"**{home_team}**"
            )

            st.dataframe(
                recent_home[
                    [
                        "Date",
                        "HomeTeam",
                        "FTHG",
                        "FTAG",
                        "AwayTeam"
                    ]
                ],
                hide_index=True,
                use_container_width=True
            )

        with c2:

            st.write(
                f"**{away_team}**"
            )

            st.dataframe(
                recent_away[
                    [
                        "Date",
                        "HomeTeam",
                        "FTHG",
                        "FTAG",
                        "AwayTeam"
                    ]
                ],
                hide_index=True,
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
        Il sistema analizza automaticamente i campionati disponibili,
        calcola le probabilità del modello e confronta tali probabilità
        con le quote presenti nei dati storici/forniti.
        """
    )

    st.info(
        "Il Value indica un vantaggio teorico del modello rispetto "
        "alla quota. Non rappresenta una certezza dell'esito."
    )

    # --------------------------------------------------------
    # FILTRI
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        min_probability = st.slider(
            "Probabilità minima",
            min_value=0.40,
            max_value=0.90,
            value=0.55,
            step=0.01
        )

    with col2:

        min_value = st.slider(
            "Value minimo",
            min_value=0.00,
            max_value=0.30,
            value=0.05,
            step=0.01,
            format="%.2f"
        )

    with col3:

        max_results = st.number_input(
            "Numero risultati",
            min_value=5,
            max_value=100,
            value=30,
            step=5
        )

    st.markdown("---")

    if st.button(
        "🚀 SCANSIONA TUTTI I CAMPIONATI",
        use_container_width=True
    ):

        with st.spinner(
            "Analizzo tutti i campionati..."
        ):

            results = scan_all_leagues(
                min_probability=
                    min_probability,

                min_value=
                    min_value
            )

        if results.empty:

            st.warning(
                "Nessuna opportunità trovata "
                "con i parametri selezionati."
            )

        else:

            results = results.head(
                max_results
            )

            # ------------------------------------------------
            # KPI
            # ------------------------------------------------

            total = len(results)

            avg_value = results[
                "Value"
            ].mean()

            avg_probability = results[
                "Probabilità"
            ].mean()

            avg_odd = results[
                "Quota"
            ].mean()

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Opportunità",
                    total
                )

            with c2:

                st.metric(
                    "Value medio",
                    f"{avg_value * 100:.1f}%"
                )

            with c3:

                st.metric(
                    "Probabilità media",
                    f"{avg_probability * 100:.1f}%"
                )

            with c4:

                st.metric(
                    "Quota media",
                    f"{avg_odd:.2f}"
                )

            st.markdown("---")

            # ------------------------------------------------
            # TABELLA
            # ------------------------------------------------

            display = results.copy()

            display[
                "Probabilità"
            ] = (
                display[
                    "Probabilità"
                ]
                * 100
            ).round(1).astype(str) + "%"

            display[
                "Value"
            ] = (
                display[
                    "Value"
                ]
                * 100
            ).round(1).astype(str) + "%"

            display[
                "Affidabilità"
            ] = (
                display[
                    "Affidabilità"
                ]
            ).round(0).astype(int).astype(str) + "%"

            display[
                "Quota"
            ] = display[
                "Quota"
            ].round(2)

            display[
                "Quota equa"
            ] = display[
                "Quota equa"
            ].round(2)

            display[
                "xG casa"
            ] = display[
                "xG casa"
            ].round(2)

            display[
                "xG ospite"
            ] = display[
                "xG ospite"
            ].round(2)

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # TOP 10
            # ------------------------------------------------

            st.markdown(
                "### 🏆 Migliori segnali del modello"
            )

            top10 = results.head(10)

            for i, (_, row) in enumerate(
                top10.iterrows(),
                start=1
            ):

                value_percent = (
                    row["Value"] * 100
                )

                probability_percent = (
                    row["Probabilità"] * 100
                )

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
                    {probability_percent:.1f}%

                    &nbsp;&nbsp;

                    <b>Quota:</b>
                    {row['Quota']:.2f}

                    &nbsp;&nbsp;

                    <b>Value:</b>
                    <span class="value-positive">
                    +{value_percent:.1f}%
                    </span>

                    <br>

                    xG:
                    {row['xG casa']:.2f}
                    -
                    {row['xG ospite']:.2f}

                    </div>
                    """,
                    unsafe_allow_html=True
                )
