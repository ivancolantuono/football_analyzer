import streamlit as st
import pandas as pd
import numpy as np
import requests
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

st.caption(
    "Analisi statistica del calcio con dati online gratuiti"
)


# ============================================================
# CONFIGURAZIONE CAMPIONATI
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

    "Austria": {
        "Bundesliga": "AUT",
    },

    "Svizzera": {
        "Super League": "SWZ",
    },

    "Danimarca": {
        "Superliga": "DNK",
    },

    "Norvegia": {
        "Eliteserien": "NOR",
    },

    "Svezia": {
        "Allsvenskan": "SWE",
    },

    "Finlandia": {
        "Veikkausliiga": "FIN",
    },

    "Irlanda": {
        "Premier Division": "IRL",
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

    "Cina": {
        "Super League": "CHN",
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
# FUNZIONE URL FOOTBALL-DATA
# ============================================================

def football_data_url(
    season,
    league_code
):

    return (
        f"https://www.football-data.co.uk/mmz4281/"
        f"{season}/{league_code}.csv"
    )


# ============================================================
# CARICAMENTO CSV
# ============================================================

@st.cache_data(ttl=3600)
def load_league(
    season,
    league_code
):

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

        if len(response.content) < 100:

            return None, "File vuoto"

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

    for column in [
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
        "AR"
    ]:

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

        return f"{str(year)[-2:]}{str(year + 1)[-2:]}"

    return f"{str(year - 1)[-2:]}{str(year)[-2:]}"


# ============================================================
# COSTRUZIONE DATAFRAME UNIFICATO
# ============================================================

def prepare_matches(
    df,
    country,
    league
):

    df = normalize_dataframe(df)

    df["Country"] = country

    df["League"] = league

    wanted = [

        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "HTHG",
        "HTAG",
        "HTR",
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
        "B365<2.5",
        "Country",
        "League"
    ]

    existing = [
        c for c in wanted
        if c in df.columns
    ]

    return df[existing].copy()


# ============================================================
# STATISTICHE SQUADRA
# ============================================================

def team_history(
    df,
    team,
    n=10
):

    if "Date" not in df.columns:

        return pd.DataFrame()

    matches = df[
        (
            df["HomeTeam"] == team
        )
        |
        (
            df["AwayTeam"] == team
        )
    ].copy()

    matches = matches.sort_values(
        "Date",
        ascending=False
    )

    return matches.head(n)


# ============================================================
# CALCOLO STATISTICHE SQUADRA
# ============================================================

def calculate_team_stats(
    df,
    team,
    n=10
):

    matches = team_history(
        df,
        team,
        n
    )

    if matches.empty:

        return None

    gf = []
    ga = []

    wins = 0
    draws = 0
    losses = 0

    over15 = 0
    over25 = 0
    over35 = 0

    btts = 0

    clean_sheet = 0

    for _, row in matches.iterrows():

        if row["HomeTeam"] == team:

            scored = row.get(
                "FTHG",
                np.nan
            )

            conceded = row.get(
                "FTAG",
                np.nan
            )

            result = row.get(
                "FTR",
                ""
            )

            if result == "H":
                wins += 1

            elif result == "D":
                draws += 1

            elif result == "A":
                losses += 1

        else:

            scored = row.get(
                "FTAG",
                np.nan
            )

            conceded = row.get(
                "FTHG",
                np.nan
            )

            result = row.get(
                "FTR",
                ""
            )

            if result == "A":
                wins += 1

            elif result == "D":
                draws += 1

            elif result == "H":
                losses += 1

        if pd.notna(scored):

            gf.append(
                float(scored)
            )

        if pd.notna(conceded):

            ga.append(
                float(conceded)
            )

        if (
            pd.notna(scored)
            and pd.notna(conceded)
        ):

            total = (
                scored
                + conceded
            )

            if total > 1.5:
                over15 += 1

            if total > 2.5:
                over25 += 1

            if total > 3.5:
                over35 += 1

            if (
                scored > 0
                and conceded > 0
            ):

                btts += 1

            if conceded == 0:

                clean_sheet += 1


    count = len(matches)

    if count == 0:

        return None

    return {

        "partite": count,

        "gol_fatti": np.mean(gf)
        if gf else 0,

        "gol_subiti": np.mean(ga)
        if ga else 0,

        "vittorie": wins,

        "pareggi": draws,

        "sconfitte": losses,

        "over15": over15 / count,

        "over25": over25 / count,

        "over35": over35 / count,

        "btts": btts / count,

        "clean_sheet": clean_sheet / count
    }


# ============================================================
# INTERFACCIA
# ============================================================

st.sidebar.header(
    "⚙️ Impostazioni"
)


# ------------------------------------------------------------
# STAGIONE
# ------------------------------------------------------------

default_season = get_current_season()

season = st.sidebar.text_input(
    "Stagione",
    value=default_season
)


# ------------------------------------------------------------
# PAESE
# ------------------------------------------------------------

countries = list(
    LEAGUES.keys()
)

country = st.sidebar.selectbox(
    "Paese",
    countries
)


# ------------------------------------------------------------
# CAMPIONATO
# ------------------------------------------------------------

league_names = list(
    LEAGUES[country].keys()
)

league = st.sidebar.selectbox(
    "Campionato",
    league_names
)


league_code = LEAGUES[
    country
][
    league
]


st.sidebar.write(
    f"Codice dati: `{league_code}`"
)


# ============================================================
# CARICAMENTO
# ============================================================

st.header(
    f"📅 {country} - {league}"
)


with st.spinner(
    "Scaricamento dati..."
):

    df_raw, error = load_league(
        season,
        league_code
    )


if error:

    st.error(
        f"❌ Errore caricamento dati: {error}"
    )

    st.info(
        "Prova un altro campionato o stagione."
    )

    st.stop()


df = prepare_matches(
    df_raw,
    country,
    league
)


# ============================================================
# INFO
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Partite",
        len(df)
    )


with col2:

    st.metric(
        "Squadre",
        len(
            set(
                df["HomeTeam"].dropna()
            )
            |
            set(
                df["AwayTeam"].dropna()
            )
        )
    )


with col3:

    if "Date" in df.columns:

        valid_dates = df[
            "Date"
        ].dropna()

        if len(valid_dates):

            st.metric(
                "Prima data",
                valid_dates.min().strftime(
                    "%d/%m/%Y"
                )
            )


with col4:

    if "Date" in df.columns:

        valid_dates = df[
            "Date"
        ].dropna()

        if len(valid_dates):

            st.metric(
                "Ultima data",
                valid_dates.max().strftime(
                    "%d/%m/%Y"
                )
            )


# ============================================================
# PARTITE
# ============================================================

st.subheader(
    "⚽ Partite"
)


display_columns = [

    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR"

]


display_columns = [
    c for c in display_columns
    if c in df.columns
]


matches_display = df[
    display_columns
].copy()


if "Date" in matches_display.columns:

    matches_display["Date"] = (
        matches_display["Date"]
        .dt.strftime("%d/%m/%Y")
    )


st.dataframe(
    matches_display,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SELEZIONE SQUADRA
# ============================================================

st.subheader(
    "📊 Analisi squadra"
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


selected_team = st.selectbox(
    "Seleziona squadra",
    teams
)


team_stats = calculate_team_stats(
    df,
    selected_team,
    n=10
)


if team_stats:

    st.markdown(
        f"### {selected_team}"
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Gol fatti",
            f"{team_stats['gol_fatti']:.2f}"
        )


    with c2:

        st.metric(
            "Gol subiti",
            f"{team_stats['gol_subiti']:.2f}"
        )


    with c3:

        st.metric(
            "Vittorie",
            team_stats["vittorie"]
        )


    with c4:

        st.metric(
            "Clean Sheet",
            f"{team_stats['clean_sheet'] * 100:.1f}%"
        )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Over 1.5",
            f"{team_stats['over15'] * 100:.1f}%"
        )


    with c2:

        st.metric(
            "Over 2.5",
            f"{team_stats['over25'] * 100:.1f}%"
        )


    with c3:

        st.metric(
            "Over 3.5",
            f"{team_stats['over35'] * 100:.1f}%"
        )


    with c4:

        st.metric(
            "Goal / Goal",
            f"{team_stats['btts'] * 100:.1f}%"
        )


# ============================================================
# ULTIME 10
# ============================================================

st.subheader(
    "📈 Ultime 10 partite"
)


history = team_history(
    df,
    selected_team,
    10
)


if not history.empty:

    history_display = history.copy()

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
        if c in history_display.columns
    ]

    history_display = history_display[
        cols
    ]

    if "Date" in history_display.columns:

        history_display["Date"] = (
            history_display["Date"]
            .dt.strftime("%d/%m/%Y")
        )

    st.dataframe(
        history_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODULO FUTURO
# ============================================================

st.divider()

st.subheader(
    "🧠 Motore probabilistico"
)

st.info(
    """
    PROSSIMO MODULO

    • Analisi di due squadre
    • Forma ultime 5/10/15
    • Casa / Trasferta
    • Media gol
    • Poisson
    • Probabilità 1X2
    • Doppia chance
    • Over / Under
    • Goal / No Goal
    • Risultato esatto
    • Confronto quote
    • Value
    • Backtest
    """
)
