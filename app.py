import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
from io import StringIO
from datetime import date


# ============================================================
# CONFIGURAZIONE
# ============================================================

st.set_page_config(
    page_title="Football Analyzer",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football Analyzer")
st.caption("Motore statistico e probabilistico")


# ============================================================
# CAMPIONATI
# ============================================================

LEAGUES = {

    "Italia": {
        "Serie A": "I1",
        "Serie B": "I2",
    },

    "Inghilterra": {
        "Premier League": "E0",
        "Championship": "E1",
        "League One": "E2",
        "League Two": "E3",
    },

    "Spagna": {
        "La Liga": "SP1",
        "La Liga 2": "SP2",
    },

    "Germania": {
        "Bundesliga": "D1",
        "2. Bundesliga": "D2",
    },

    "Francia": {
        "Ligue 1": "F1",
        "Ligue 2": "F2",
    },

    "Olanda": {
        "Eredivisie": "N1",
    },

    "Belgio": {
        "Jupiler League": "B1",
    },

    "Portogallo": {
        "Primeira Liga": "P1",
    },

    "Turchia": {
        "Super Lig": "T1",
    },

    "Grecia": {
        "Super League": "G1",
    },

    "Scozia": {
        "Premiership": "SC0",
        "Championship": "SC1",
        "League One": "SC2",
        "League Two": "SC3",
    },

    "Polonia": {
        "Ekstraklasa": "POL",
    },

    "Romania": {
        "Liga 1": "ROM",
    },

    "Russia": {
        "Premier League": "RUS",
    },

    "USA": {
        "MLS": "USA",
    },

    "Giappone": {
        "J League": "JPN",
    },

    "Brasile": {
        "Serie A": "BRA",
    },

    "Argentina": {
        "Primera Division": "ARG",
    },

    "Messico": {
        "Liga MX": "MEX",
    },
}


# ============================================================
# URL FOOTBALL DATA
# ============================================================

def football_data_url(season, league_code):

    return (
        f"https://www.football-data.co.uk/mmz4281/"
        f"{season}/{league_code}.csv"
    )


# ============================================================
# DOWNLOAD DATI
# ============================================================

@st.cache_data(ttl=3600)
def load_league(season, league_code):

    url = football_data_url(
        season,
        league_code
    )

    try:

        response = requests.get(
            url,
            timeout=30
        )

        if response.status_code != 200:

            return None, (
                f"HTTP {response.status_code}"
            )

        df = pd.read_csv(
            StringIO(
                response.content.decode(
                    "latin1"
                )
            )
        )

        return df, None

    except Exception as e:

        return None, str(e)


# ============================================================
# NORMALIZZAZIONE
# ============================================================

def normalize_dataframe(df):

    df = df.copy()

    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"],
            dayfirst=True,
            errors="coerce"
        )

    numeric_columns = [
        "FTHG",
        "FTAG",
        "HTHG",
        "HTAG",
        "HS",
        "AS",
        "HST",
        "AST",
        "HC",
        "AC",
        "HF",
        "AF",
        "HY",
        "AY",
        "HR",
        "AR",
        "B365H",
        "B365D",
        "B365A",
        "B365>2.5",
        "B365<2.5"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# STAGIONE
# ============================================================

def get_current_season():

    today = date.today()

    year = today.year

    if today.month >= 7:

        return (
            f"{str(year)[-2:]}"
            f"{str(year + 1)[-2:]}"
        )

    return (
        f"{str(year - 1)[-2:]}"
        f"{str(year)[-2:]}"
    )


# ============================================================
# SOLO PARTITE CON RISULTATO
# ============================================================

def played_matches(df):

    required = [
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG"
    ]

    existing = [
        c for c in required
        if c in df.columns
    ]

    if len(existing) < 4:

        return pd.DataFrame()

    result = df.dropna(
        subset=[
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG"
        ]
    ).copy()

    if "Date" in result.columns:

        result = result.sort_values(
            "Date"
        )

    return result


# ============================================================
# MEDIA CAMPIONATO
# ============================================================

def league_averages(df):

    matches = played_matches(df)

    if matches.empty:

        return {
            "home_goals": 1.40,
            "away_goals": 1.10
        }

    home_avg = matches["FTHG"].mean()
    away_avg = matches["FTAG"].mean()

    return {
        "home_goals": float(home_avg),
        "away_goals": float(away_avg)
    }


# ============================================================
# STORICO SQUADRA
# ============================================================

def team_matches(
    df,
    team,
    venue=None,
    n=None
):

    matches = played_matches(df)

    matches = matches[
        (
            matches["HomeTeam"] == team
        )
        |
        (
            matches["AwayTeam"] == team
        )
    ].copy()

    if venue == "home":

        matches = matches[
            matches["HomeTeam"] == team
        ]

    elif venue == "away":

        matches = matches[
            matches["AwayTeam"] == team
        ]

    matches = matches.sort_values(
        "Date",
        ascending=False
    )

    if n:

        matches = matches.head(n)

    return matches


# ============================================================
# STATISTICHE SQUADRA
# ============================================================

def team_stats(
    df,
    team,
    venue=None,
    n=10
):

    matches = team_matches(
        df,
        team,
        venue,
        n
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
    over45 = 0

    btts = 0
    clean = 0
    failed_to_score = 0

    for _, row in matches.iterrows():

        if row["HomeTeam"] == team:

            gf = float(row["FTHG"])
            ga = float(row["FTAG"])

            if gf > ga:
                wins += 1

            elif gf == ga:
                draws += 1

            else:
                losses += 1

        else:

            gf = float(row["FTAG"])
            ga = float(row["FTHG"])

            if gf > ga:
                wins += 1

            elif gf == ga:
                draws += 1

            else:
                losses += 1

        goals_for.append(gf)
        goals_against.append(ga)

        total = gf + ga

        if total > 0.5:
            over05 += 1

        if total > 1.5:
            over15 += 1

        if total > 2.5:
            over25 += 1

        if total > 3.5:
            over35 += 1

        if total > 4.5:
            over45 += 1

        if gf > 0 and ga > 0:
            btts += 1

        if ga == 0:
            clean += 1

        if gf == 0:
            failed_to_score += 1

    count = len(matches)

    return {

        "matches": count,

        "gf": np.mean(goals_for),
        "ga": np.mean(goals_against),

        "wins": wins,
        "draws": draws,
        "losses": losses,

        "over05": over05 / count,
        "over15": over15 / count,
        "over25": over25 / count,
        "over35": over35 / count,
        "over45": over45 / count,

        "btts": btts / count,

        "clean": clean / count,

        "failed_score": failed_to_score / count
    }


# ============================================================
# POISSON
# ============================================================

def poisson_probability(
    goals,
    expected_goals
):

    return (
        math.exp(-expected_goals)
        *
        expected_goals ** goals
        /
        math.factorial(goals)
    )


# ============================================================
# DISTRIBUZIONE RISULTATI
# ============================================================

def score_matrix(
    home_lambda,
    away_lambda,
    max_goals=8
):

    matrix = np.zeros(
        (
            max_goals + 1,
            max_goals + 1
        )
    )

    for home in range(
        max_goals + 1
    ):

        for away in range(
            max_goals + 1
        ):

            matrix[home, away] = (
                poisson_probability(
                    home,
                    home_lambda
                )
                *
                poisson_probability(
                    away,
                    away_lambda
                )
            )

    total = matrix.sum()

    if total > 0:

        matrix = matrix / total

    return matrix


# ============================================================
# CALCOLO PROBABILITÀ
# ============================================================

def calculate_probabilities(
    matrix
):

    max_goals = matrix.shape[0] - 1

    home_win = 0
    draw = 0
    away_win = 0

    btts = 0

    home_scores = 0
    away_scores = 0

    home_clean = 0
    away_clean = 0

    over = {
        0.5: 0,
        1.5: 0,
        2.5: 0,
        3.5: 0,
        4.5: 0
    }

    exact_scores = []

    for home in range(
        max_goals + 1
    ):

        for away in range(
            max_goals + 1
        ):

            p = matrix[
                home,
                away
            ]

            total = home + away

            if home > away:

                home_win += p

            elif home == away:

                draw += p

            else:

                away_win += p

            if home > 0:

                home_scores += p

            if away > 0:

                away_scores += p

            if away == 0:

                home_clean += p

            if home == 0:

                away_clean += p

            if home > 0 and away > 0:

                btts += p

            for line in over:

                if total > line:

                    over[line] += p

            exact_scores.append(
                (
                    home,
                    away,
                    p
                )
            )

    exact_scores = sorted(
        exact_scores,
        key=lambda x: x[2],
        reverse=True
    )

    return {

        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,

        "double_1x": home_win + draw,
        "double_x2": draw + away_win,
        "double_12": home_win + away_win,

        "btts": btts,

        "home_scores": home_scores,
        "away_scores": away_scores,

        "home_clean": home_clean,
        "away_clean": away_clean,

        "over": over,

        "under": {
            line: 1 - probability
            for line, probability
            in over.items()
        },

        "exact_scores": exact_scores
    }


# ============================================================
# FORZA SQUADRA
# ============================================================

def calculate_expected_goals(
    df,
    home_team,
    away_team,
    n=10
):

    averages = league_averages(df)

    home_stats = team_stats(
        df,
        home_team,
        venue="home",
        n=n
    )

    away_stats = team_stats(
        df,
        away_team,
        venue="away",
        n=n
    )

    # --------------------------------------------------------
    # Se non ci sono abbastanza dati
    # --------------------------------------------------------

    if home_stats is None:

        home_stats = team_stats(
            df,
            home_team,
            venue=None,
            n=n
        )

    if away_stats is None:

        away_stats = team_stats(
            df,
            away_team,
            venue=None,
            n=n
        )

    if home_stats is None:

        home_stats = {
            "gf": averages["home_goals"],
            "ga": averages["away_goals"],
            "matches": 0
        }

    if away_stats is None:

        away_stats = {
            "gf": averages["away_goals"],
            "ga": averages["home_goals"],
            "matches": 0
        }

    # --------------------------------------------------------
    # Forza attacco
    # --------------------------------------------------------

    home_attack = (
        home_stats["gf"]
        /
        averages["home_goals"]
        if averages["home_goals"] > 0
        else 1
    )

    away_attack = (
        away_stats["gf"]
        /
        averages["away_goals"]
        if averages["away_goals"] > 0
        else 1
    )

    # --------------------------------------------------------
    # Forza difesa
    #
    # 1 = difesa media
    # <1 = difesa migliore
    # >1 = difesa peggiore
    # --------------------------------------------------------

    home_defence = (
        home_stats["ga"]
        /
        averages["away_goals"]
        if averages["away_goals"] > 0
        else 1
    )

    away_defence = (
        away_stats["ga"]
        /
        averages["home_goals"]
        if averages["home_goals"] > 0
        else 1
    )

    # --------------------------------------------------------
    # Gol attesi
    # --------------------------------------------------------

    expected_home = (
        averages["home_goals"]
        *
        home_attack
        *
        away_defence
    )

    expected_away = (
        averages["away_goals"]
        *
        away_attack
        *
        home_defence
    )

    # --------------------------------------------------------
    # Limiti di sicurezza
    # --------------------------------------------------------

    expected_home = max(
        0.05,
        min(expected_home, 5.0)
    )

    expected_away = max(
        0.05,
        min(expected_away, 5.0)
    )

    return (
        expected_home,
        expected_away,
        home_stats,
        away_stats,
        averages
    )


# ============================================================
# FUNZIONE PERCENTUALE
# ============================================================

def pct(value):

    return f"{value * 100:.1f}%"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Impostazioni"
)


season = st.sidebar.text_input(
    "Stagione",
    value=get_current_season()
)


country = st.sidebar.selectbox(
    "Paese",
    list(LEAGUES.keys())
)


league = st.sidebar.selectbox(
    "Campionato",
    list(
        LEAGUES[country].keys()
    )
)


league_code = LEAGUES[
    country
][
    league
]


n_matches = st.sidebar.select_slider(
    "Partite utilizzate",
    options=[5, 10, 15],
    value=10
)


# ============================================================
# DOWNLOAD
# ============================================================

st.header(
    f"📊 {country} - {league}"
)


with st.spinner(
    "Scaricamento dati..."
):

    df, error = load_league(
        season,
        league_code
    )


if error:

    st.error(
        f"Errore caricamento: {error}"
    )

    st.stop()


df = normalize_dataframe(df)

played = played_matches(df)


# ============================================================
# INFO CAMPIONATO
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Partite disponibili",
        len(played)
    )


with col2:

    st.metric(
        "Squadre",
        len(
            set(
                played["HomeTeam"]
            )
            |
            set(
                played["AwayTeam"]
            )
        )
    )


with col3:

    averages = league_averages(
        df
    )

    st.metric(
        "Media gol partita",
        f"{averages['home_goals'] + averages['away_goals']:.2f}"
    )


# ============================================================
# SQUADRE
# ============================================================

teams = sorted(
    list(
        set(
            played["HomeTeam"]
        )
        |
        set(
            played["AwayTeam"]
        )
    )
)


st.divider()

st.header(
    "⚽ Analizza una partita"
)


if len(teams) < 2:

    st.warning(
        "Non ci sono abbastanza squadre."
    )

    st.stop()


col1, col2 = st.columns(2)


with col1:

    home_team = st.selectbox(
        "🏠 Squadra casa",
        teams
    )


with col2:

    away_options = [
        team
        for team in teams
        if team != home_team
    ]

    away_team = st.selectbox(
        "✈️ Squadra ospite",
        away_options
    )


# ============================================================
# CALCOLO MODELLO
# ============================================================

try:

    (
        expected_home,
        expected_away,
        home_stats,
        away_stats,
        averages
    ) = calculate_expected_goals(
        df,
        home_team,
        away_team,
        n_matches
    )

    matrix = score_matrix(
        expected_home,
        expected_away
    )

    probabilities = calculate_probabilities(
        matrix
    )

except Exception as e:

    st.error(
        "Errore nel calcolo del modello."
    )

    st.exception(e)

    st.stop()


# ============================================================
# GOL ATTESI
# ============================================================

st.divider()

st.header(
    "🎯 Gol attesi"
)


c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "Gol attesi casa",
        f"{expected_home:.2f}"
    )


with c2:

    st.metric(
        "Gol attesi ospite",
        f"{expected_away:.2f}"
    )


with c3:

    st.metric(
        "Gol attesi totali",
        f"{expected_home + expected_away:.2f}"
    )


# ============================================================
# 1X2
# ============================================================

st.divider()

st.header(
    "🏆 Probabilità 1X2"
)


c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "1 - Casa",
        pct(
            probabilities["home_win"]
        )
    )


with c2:

    st.metric(
        "X - Pareggio",
        pct(
            probabilities["draw"]
        )
    )


with c3:

    st.metric(
        "2 - Ospite",
        pct(
            probabilities["away_win"]
        )
    )


# ============================================================
# DOPPIA CHANCE
# ============================================================

st.subheader(
    "Doppia chance"
)


c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "1X",
        pct(
            probabilities["double_1x"]
        )
    )


with c2:

    st.metric(
        "X2",
        pct(
            probabilities["double_x2"]
        )
    )


with c3:

    st.metric(
        "12",
        pct(
            probabilities["double_12"]
        )
    )


# ============================================================
# OVER / UNDER
# ============================================================

st.divider()

st.header(
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

        "Linea": line,

        "Over":
            pct(
                probabilities[
                    "over"
                ][line]
            ),

        "Under":
            pct(
                probabilities[
                    "under"
                ][line]
            )
    })


st.dataframe(
    pd.DataFrame(rows),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# GOAL / NOGOAL
# ============================================================

st.divider()

st.header(
    "🥅 Goal / No Goal"
)


c1, c2 = st.columns(2)


with c1:

    st.metric(
        "GOAL",
        pct(
            probabilities["btts"]
        )
    )


with c2:

    st.metric(
        "NO GOAL",
        pct(
            1 -
            probabilities["btts"]
        )
    )


# ============================================================
# GOL SQUADRE
# ============================================================

st.subheader(
    "Gol squadra"
)


c1, c2 = st.columns(2)


with c1:

    st.write(
        f"**{home_team}**"
    )

    st.metric(
        "Segna almeno 1 gol",
        pct(
            probabilities[
                "home_scores"
            ]
        )
    )

    st.metric(
        "Clean sheet",
        pct(
            probabilities[
                "home_clean"
            ]
        )
    )


with c2:

    st.write(
        f"**{away_team}**"
    )

    st.metric(
        "Segna almeno 1 gol",
        pct(
            probabilities[
                "away_scores"
            ]
        )
    )

    st.metric(
        "Clean sheet",
        pct(
            probabilities[
                "away_clean"
            ]
        )
    )


# ============================================================
# RISULTATI ESATTI
# ============================================================

st.divider()

st.header(
    "🎯 Risultati esatti più probabili"
)


exact_rows = []


for home, away, probability in (
    probabilities["exact_scores"][:10]
):

    exact_rows.append({

        "Risultato":
            f"{home}-{away}",

        "Probabilità":
            pct(probability)
    })


st.dataframe(
    pd.DataFrame(exact_rows),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# STATISTICHE SQUADRE
# ============================================================

st.divider()

st.header(
    "📈 Statistiche utilizzate dal modello"
)


if home_stats and away_stats:

    comparison = pd.DataFrame({

        home_team: {

            "Partite":
                home_stats["matches"],

            "Gol fatti / partita":
                round(
                    home_stats["gf"],
                    2
                ),

            "Gol subiti / partita":
                round(
                    home_stats["ga"],
                    2
                ),

            "Over 1.5":
                pct(
                    home_stats["over15"]
                ),

            "Over 2.5":
                pct(
                    home_stats["over25"]
                ),

            "Over 3.5":
                pct(
                    home_stats["over35"]
                ),

            "Goal":
                pct(
                    home_stats["btts"]
                ),

            "Clean sheet":
                pct(
                    home_stats["clean"]
                ),

            "Non segna":
                pct(
                    home_stats["failed_score"]
                )
        },

        away_team: {

            "Partite":
                away_stats["matches"],

            "Gol fatti / partita":
                round(
                    away_stats["gf"],
                    2
                ),

            "Gol subiti / partita":
                round(
                    away_stats["ga"],
                    2
                ),

            "Over 1.5":
                pct(
                    away_stats["over15"]
                ),

            "Over 2.5":
                pct(
                    away_stats["over25"]
                ),

            "Over 3.5":
                pct(
                    away_stats["over35"]
                ),

            "Goal":
                pct(
                    away_stats["btts"]
                ),

            "Clean sheet":
                pct(
                    away_stats["clean"]
                ),

            "Non segna":
                pct(
                    away_stats["failed_score"]
                )
        }
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )


# ============================================================
# ULTIME PARTITE
# ============================================================

st.divider()

st.header(
    "📋 Ultime partite"
)


tab1, tab2 = st.tabs([
    f"🏠 {home_team}",
    f"✈️ {away_team}"
])


with tab1:

    h = team_matches(
        df,
        home_team,
        venue=None,
        n=n_matches
    )

    if not h.empty:

        cols = [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR"
        ]

        cols = [
            c for c in cols
            if c in h.columns
        ]

        st.dataframe(
            h[cols],
            use_container_width=True,
            hide_index=True
        )


with tab2:

    a = team_matches(
        df,
        away_team,
        venue=None,
        n=n_matches
    )

    if not a.empty:

        cols = [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR"
        ]

        cols = [
            c for c in cols
            if c in a.columns
        ]

        st.dataframe(
            a[cols],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# SPIEGAZIONE MODELLO
# ============================================================

st.divider()

with st.expander(
    "🧠 Come viene calcolata la probabilità?"
):

    st.write(
        """
        Il modello utilizza una distribuzione di Poisson.

        Per prima cosa vengono calcolate:

        • media gol del campionato
        • gol fatti dalla squadra casa
        • gol subiti dalla squadra casa
        • gol fatti dalla squadra ospite
        • gol subiti dalla squadra ospite

        Viene poi stimata la forza offensiva e difensiva
        delle due squadre.

        Da queste informazioni vengono stimati i gol attesi
        della squadra casa e della squadra ospite.

        La distribuzione di Poisson permette quindi di
        ottenere la probabilità di ogni risultato possibile.

        Dalla matrice dei risultati vengono ricavate:

        • 1X2
        • doppia chance
        • Over / Under
        • Goal / No Goal
        • gol delle singole squadre
        • clean sheet
        • risultati esatti
        """
    )


# ============================================================
# NOTA
# ============================================================

st.info(
    """
    ⚠️ Le percentuali sono stime statistiche del modello.
    Non rappresentano una certezza sull'esito della partita.
    """
)
