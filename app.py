import streamlit as st
import pandas as pd
from datetime import date, datetime

from curl_cffi import requests


# ============================================================
# CONFIGURAZIONE
# ============================================================

st.set_page_config(
    page_title="Football Analyzer",
    page_icon="⚽",
    layout="wide"
)

BASE_URL = "https://api.sofascore.com/api/v1"


# ============================================================
# SESSIONE SOFASCORE
# ============================================================

@st.cache_resource
def get_session():

    session = requests.Session(
        impersonate="chrome"
    )

    session.headers.update({
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.sofascore.com/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/140.0.0.0 Safari/537.36"
        )
    })

    return session


# ============================================================
# FUNZIONE GENERALE SOFASCORE
# ============================================================

def sofascore_get(endpoint):

    url = BASE_URL + endpoint

    try:

        session = get_session()

        response = session.get(
            url,
            timeout=20
        )

        if response.status_code != 200:

            return None, response.status_code

        return response.json(), 200

    except Exception as e:

        return None, str(e)


# ============================================================
# PARTITE DEL GIORNO
# ============================================================

@st.cache_data(ttl=300)
def get_events(selected_date):

    endpoint = (
        f"/sport/football/scheduled-events/"
        f"{selected_date}"
    )

    return sofascore_get(endpoint)


# ============================================================
# DETTAGLI PARTITA
# ============================================================

@st.cache_data(ttl=300)
def get_event_details(event_id):

    return sofascore_get(
        f"/event/{event_id}"
    )


# ============================================================
# STATISTICHE PARTITA
# ============================================================

@st.cache_data(ttl=300)
def get_event_statistics(event_id):

    return sofascore_get(
        f"/event/{event_id}/statistics"
    )


# ============================================================
# FUNZIONI DI SUPPORTO
# ============================================================

def get_team_name(team):

    if not team:
        return "N/D"

    return team.get(
        "name",
        "N/D"
    )


def get_tournament_name(event):

    tournament = event.get(
        "tournament",
        {}
    )

    return tournament.get(
        "name",
        "N/D"
    )


def get_country_name(event):

    tournament = event.get(
        "tournament",
        {}
    )

    category = tournament.get(
        "category",
        {}
    )

    return category.get(
        "name",
        "N/D"
    )


def get_match_datetime(event):

    timestamp = event.get(
        "startTimestamp"
    )

    if not timestamp:
        return "N/D"

    try:

        dt = datetime.fromtimestamp(
            timestamp
        )

        return dt.strftime(
            "%H:%M"
        )

    except Exception:

        return "N/D"


def get_match_status(event):

    status = event.get(
        "status",
        {}
    )

    return status.get(
        "description",
        "N/D"
    )


def get_score(event):

    home_score = event.get(
        "homeScore",
        {}
    )

    away_score = event.get(
        "awayScore",
        {}
    )

    home = home_score.get(
        "current"
    )

    away = away_score.get(
        "current"
    )

    if home is None or away is None:

        return "-"

    return f"{home} - {away}"


# ============================================================
# ESTRAZIONE PARTITE
# ============================================================

def build_matches_dataframe(events):

    rows = []

    for event in events:

        home_team = event.get(
            "homeTeam",
            {}
        )

        away_team = event.get(
            "awayTeam",
            {}
        )

        rows.append({

            "ID": event.get(
                "id"
            ),

            "Ora": get_match_datetime(
                event
            ),

            "Paese": get_country_name(
                event
            ),

            "Campionato": get_tournament_name(
                event
            ),

            "Casa": get_team_name(
                home_team
            ),

            "Trasferta": get_team_name(
                away_team
            ),

            "Stato": get_match_status(
                event
            ),

            "Risultato": get_score(
                event
            )
        })

    if not rows:

        return pd.DataFrame()

    df = pd.DataFrame(rows)

    return df


# ============================================================
# STATISTICHE
# ============================================================

def parse_statistics(data):

    if not data:
        return pd.DataFrame()

    rows = []

    periods = data.get(
        "statistics",
        []
    )

    for period in periods:

        period_name = period.get(
            "period",
            "N/D"
        )

        groups = period.get(
            "groups",
            []
        )

        for group in groups:

            group_name = group.get(
                "groupName",
                "N/D"
            )

            statistics_items = group.get(
                "statisticsItems",
                []
            )

            for item in statistics_items:

                rows.append({

                    "Periodo": period_name,

                    "Categoria": group_name,

                    "Statistica": item.get(
                        "name",
                        "N/D"
                    ),

                    "Casa": item.get(
                        "home",
                        ""
                    ),

                    "Trasferta": item.get(
                        "away",
                        ""
                    )
                })

    return pd.DataFrame(rows)


# ============================================================
# TITOLO
# ============================================================

st.title(
    "⚽ Football Analyzer"
)

st.markdown(
    """
    ### Analisi statistica delle partite di calcio

    Dati raccolti online e utilizzati per costruire
    successivamente le probabilità dei vari mercati.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Impostazioni"
)

selected_date = st.sidebar.date_input(
    "Data partite",
    value=date.today()
)

if st.sidebar.button(
    "🔄 Aggiorna dati"
):

    st.cache_data.clear()

    st.rerun()


# ============================================================
# CARICAMENTO PARTITE
# ============================================================

with st.spinner(
    "Recupero partite da SofaScore..."
):

    data, status = get_events(
        selected_date.strftime(
            "%Y-%m-%d"
        )
    )


# ============================================================
# GESTIONE ERRORE
# ============================================================

if data is None:

    st.error(
        "❌ Non riesco a recuperare "
        "i dati da SofaScore."
    )

    st.write(
        f"Risposta HTTP: {status}"
    )

    st.info(
        """
        Se compare ancora 403, il server che
        ospita Streamlit potrebbe essere bloccato
        dal sistema di protezione di SofaScore.
        """
    )

    st.stop()


# ============================================================
# ESTRAZIONE EVENTI
# ============================================================

events = data.get(
    "events",
    []
)

if not events:

    st.warning(
        "Nessuna partita trovata per questa data."
    )

    st.stop()


# ============================================================
# DATAFRAME
# ============================================================

df = build_matches_dataframe(
    events
)


# ============================================================
# FILTRI
# ============================================================

st.subheader(
    "🔎 Filtri"
)

col1, col2 = st.columns(2)


with col1:

    countries = sorted(
        df["Paese"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_country = st.selectbox(
        "Paese",
        ["Tutti"] + countries
    )


with col2:

    tournaments = sorted(
        df["Campionato"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_tournament = st.selectbox(
        "Campionato",
        ["Tutti"] + tournaments
    )


filtered_df = df.copy()


if selected_country != "Tutti":

    filtered_df = filtered_df[
        filtered_df["Paese"]
        == selected_country
    ]


if selected_tournament != "Tutti":

    filtered_df = filtered_df[
        filtered_df["Campionato"]
        == selected_tournament
    ]


# ============================================================
# NUMERO PARTITE
# ============================================================

st.metric(
    "Partite trovate",
    len(filtered_df)
)


# ============================================================
# TABELLA
# ============================================================

st.subheader(
    "📋 Partite"
)

display_columns = [

    "Ora",
    "Paese",
    "Campionato",
    "Casa",
    "Trasferta",
    "Stato",
    "Risultato"

]

st.dataframe(
    filtered_df[
        display_columns
    ],
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SELEZIONE PARTITA
# ============================================================

st.subheader(
    "⚽ Analizza partita"
)

if len(filtered_df) > 0:

    match_options = {}

    for _, row in filtered_df.iterrows():

        label = (
            f"{row['Ora']} | "
            f"{row['Casa']} - "
            f"{row['Trasferta']} | "
            f"{row['Campionato']}"
        )

        match_options[label] = row["ID"]


    selected_match = st.selectbox(
        "Seleziona partita",
        list(match_options.keys())
    )


    event_id = match_options[
        selected_match
    ]


    # ========================================================
    # DETTAGLI
    # ========================================================

    with st.spinner(
        "Recupero dettagli partita..."
    ):

        details, details_status = (
            get_event_details(
                event_id
            )
        )


    if details is None:

        st.warning(
            f"Impossibile recuperare "
            f"i dettagli. Risposta: "
            f"{details_status}"
        )

    else:

        event = details.get(
            "event",
            {}
        )

        home_team = get_team_name(
            event.get(
                "homeTeam",
                {}
            )
        )

        away_team = get_team_name(
            event.get(
                "awayTeam",
                {}
            )
        )

        score = get_score(
            event
        )


        st.markdown(
            f"""
            ## {home_team} 🆚 {away_team}

            ### Risultato: **{score}**
            """
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.write(
                "**Campionato**"
            )

            st.write(
                get_tournament_name(
                    event
                )
            )


        with col2:

            st.write(
                "**Paese**"
            )

            st.write(
                get_country_name(
                    event
                )
            )


        with col3:

            st.write(
                "**Stato**"
            )

            st.write(
                get_match_status(
                    event
                )
            )


    # ========================================================
    # STATISTICHE
    # ========================================================

    st.subheader(
        "📊 Statistiche"
    )

    with st.spinner(
        "Recupero statistiche..."
    ):

        statistics, statistics_status = (
            get_event_statistics(
                event_id
            )
        )


    if statistics is None:

        st.warning(
            f"Statistiche non disponibili. "
            f"Risposta: {statistics_status}"
        )

    else:

        stats_df = parse_statistics(
            statistics
        )

        if stats_df.empty:

            st.info(
                "Nessuna statistica disponibile."
            )

        else:

            st.dataframe(
                stats_df,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# FUTURO MODULO PROBABILITÀ
# ============================================================

st.divider()

st.subheader(
    "🧠 Modulo probabilità"
)

st.info(
    """
    Prossimo modulo:

    • Probabilità 1X2
    • Doppia chance
    • Over / Under
    • Goal / No Goal
    • Over 0.5
    • Over 1.5
    • Over 2.5
    • Over 3.5
    • Over 4.5
    • Gol casa
    • Gol trasferta
    • Clean sheet
    • Risultato esatto
    • Probabilità primo tempo
    • Probabilità secondo tempo
    • Forma recente
    • Media gol fatti/subiti
    • Statistiche casa/trasferta
    """
)
