import streamlit as st
import requests
import pandas as pd
from datetime import datetime, date

# ============================================================
# CONFIGURAZIONE
# ============================================================

st.set_page_config(
    page_title="Football Analyzer",
    page_icon="⚽",
    layout="wide"
)

BASE_URL = "https://www.sofascore.com/api/v1"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.sofascore.com/",
}


# ============================================================
# FUNZIONI SOFASCORE
# ============================================================

def sofascore_get(endpoint):

    url = BASE_URL + endpoint

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20
        )

        if response.status_code != 200:
            return None, response.status_code

        return response.json(), 200

    except requests.exceptions.RequestException as e:

        return None, str(e)

    except Exception as e:

        return None, str(e)


@st.cache_data(ttl=300)
def get_events(selected_date):

    endpoint = (
        f"/sport/football/scheduled-events/"
        f"{selected_date}"
    )

    return sofascore_get(endpoint)


@st.cache_data(ttl=300)
def get_event_details(event_id):

    endpoint = f"/event/{event_id}"

    return sofascore_get(endpoint)


@st.cache_data(ttl=300)
def get_event_statistics(event_id):

    endpoint = f"/event/{event_id}/statistics"

    return sofascore_get(endpoint)


# ============================================================
# FUNZIONI DI SUPPORTO
# ============================================================

def get_team_name(team):

    if not team:
        return "?"

    return team.get("name", "?")


def get_tournament_name(event):

    tournament = event.get("tournament", {})

    return tournament.get(
        "name",
        "Campionato sconosciuto"
    )


def get_country_name(event):

    tournament = event.get("tournament", {})

    category = tournament.get(
        "category",
        {}
    )

    return category.get(
        "name",
        "?"
    )


def get_event_time(event):

    timestamp = event.get("startTimestamp")

    if not timestamp:
        return "?"

    try:

        return datetime.fromtimestamp(
            timestamp
        ).strftime("%H:%M")

    except:

        return "?"


def get_status(event):

    status = event.get(
        "status",
        {}
    )

    return status.get(
        "description",
        "?"
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
# HEADER
# ============================================================

st.title("⚽ Football Analyzer")

st.markdown(
    "### Analisi automatica delle partite"
)

st.caption(
    "Sorgente dati: Sofascore • "
    "nessuna API key richiesta"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Filtri")

selected_date = st.sidebar.date_input(
    "📅 Data",
    value=date.today()
)

if st.sidebar.button(
    "🔄 Aggiorna dati",
    use_container_width=True
):

    st.cache_data.clear()
    st.rerun()


# ============================================================
# CARICAMENTO PARTITE
# ============================================================

with st.spinner(
    "Recupero partite da Sofascore..."
):

    data, status_code = get_events(
        selected_date.isoformat()
    )


if data is None:

    st.error(
        "❌ Non riesco a recuperare i dati da Sofascore."
    )

    st.info(
        f"Risposta HTTP: {status_code}"
    )

    st.stop()


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
# CREAZIONE DATAFRAME
# ============================================================

rows = []

for event in events:

    home = get_team_name(
        event.get("homeTeam")
    )

    away = get_team_name(
        event.get("awayTeam")
    )

    tournament = get_tournament_name(
        event
    )

    country = get_country_name(
        event
    )

    rows.append({

        "ID": event.get("id"),

        "Ora": get_event_time(
            event
        ),

        "Nazione": country,

        "Campionato": tournament,

        "Casa": home,

        "Trasferta": away,

        "Stato": get_status(
            event
        ),

        "Risultato": get_score(
            event
        )

    })


df = pd.DataFrame(rows)


# ============================================================
# FILTRI
# ============================================================

st.sidebar.markdown("---")

countries = sorted(
    df["Nazione"]
    .dropna()
    .unique()
    .tolist()
)

selected_country = st.sidebar.selectbox(
    "🌍 Nazione",
    ["Tutte"] + countries
)


filtered = df.copy()


if selected_country != "Tutte":

    filtered = filtered[
        filtered["Nazione"] ==
        selected_country
    ]


tournaments = sorted(
    filtered["Campionato"]
    .dropna()
    .unique()
    .tolist()
)


selected_tournament = st.sidebar.selectbox(
    "🏆 Campionato",
    ["Tutti"] + tournaments
)


if selected_tournament != "Tutti":

    filtered = filtered[
        filtered["Campionato"] ==
        selected_tournament
    ]


# ============================================================
# RIEPILOGO
# ============================================================

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Partite trovate",
        len(filtered)
    )

with c2:

    st.metric(
        "Campionati",
        filtered["Campionato"].nunique()
    )

with c3:

    st.metric(
        "Nazioni",
        filtered["Nazione"].nunique()
    )


st.markdown("---")


# ============================================================
# ELENCO PARTITE
# ============================================================

st.subheader(
    f"📅 Partite del {selected_date.strftime('%d/%m/%Y')}"
)


display_df = filtered[
    [
        "Ora",
        "Nazione",
        "Campionato",
        "Casa",
        "Trasferta",
        "Stato",
        "Risultato"
    ]
]


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SELEZIONE PARTITA
# ============================================================

st.markdown("---")

st.subheader(
    "🔎 Analizza una partita"
)


if len(filtered) == 0:

    st.warning(
        "Nessuna partita corrisponde ai filtri."
    )

    st.stop()


match_options = {}

for _, row in filtered.iterrows():

    label = (
        f"{row['Ora']} | "
        f"{row['Campionato']} | "
        f"{row['Casa']} - "
        f"{row['Trasferta']}"
    )

    match_options[label] = int(
        row["ID"]
    )


selected_match = st.selectbox(
    "Seleziona partita",
    list(match_options.keys())
)


event_id = match_options[
    selected_match
]


# ============================================================
# DETTAGLIO PARTITA
# ============================================================

with st.spinner(
    "Recupero dettagli partita..."
):

    event_data, event_status = (
        get_event_details(event_id)
    )


if event_data is None:

    st.error(
        f"Impossibile recuperare il dettaglio "
        f"della partita. HTTP: {event_status}"
    )

else:

    event = event_data.get(
        "event",
        event_data
    )

    home_team = get_team_name(
        event.get("homeTeam")
    )

    away_team = get_team_name(
        event.get("awayTeam")
    )


    st.markdown("---")

    st.subheader(
        f"⚽ {home_team}  -  {away_team}"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Campionato",
            get_tournament_name(event)
        )


    with col2:

        st.metric(
            "Ora",
            get_event_time(event)
        )


    with col3:

        st.metric(
            "Stato",
            get_status(event)
        )


    # --------------------------------------------------------
    # RISULTATO
    # --------------------------------------------------------

    home_score = event.get(
        "homeScore",
        {}
    )

    away_score = event.get(
        "awayScore",
        {}
    )

    score_home = home_score.get(
        "current",
        "-"
    )

    score_away = away_score.get(
        "current",
        "-"
    )


    st.markdown(
        f"## {score_home}  -  {score_away}"
    )


    # --------------------------------------------------------
    # STATISTICHE
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "📊 Statistiche partita"
    )


    with st.spinner(
        "Recupero statistiche..."
    ):

        stats_data, stats_status = (
            get_event_statistics(event_id)
        )


    if stats_data is None:

        st.info(
            "Le statistiche dettagliate "
            "non sono disponibili per questa partita."
        )

    else:

        statistics = stats_data.get(
            "statistics",
            []
        )


        if not statistics:

            st.info(
                "Nessuna statistica disponibile."
            )

        else:

            for period in statistics:

                period_name = period.get(
                    "period",
                    "ALL"
                )

                if period_name == "ALL":

                    title = "Partita"

                elif period_name == "1ST":

                    title = "Primo tempo"

                elif period_name == "2ND":

                    title = "Secondo tempo"

                else:

                    title = period_name


                with st.expander(
                    f"📊 {title}",
                    expanded=(
                        period_name == "ALL"
                    )
                ):

                    groups = period.get(
                        "groups",
                        []
                    )


                    for group in groups:

                        group_name = group.get(
                            "groupName",
                            "Statistiche"
                        )

                        st.markdown(
                            f"**{group_name}**"
                        )


                        stat_rows = []


                        for item in group.get(
                            "statisticsItems",
                            []
                        ):

                            stat_rows.append({

                                "Statistica":
                                    item.get(
                                        "name",
                                        "?"
                                    ),

                                home_team:
                                    item.get(
                                        "home",
                                        "-"
                                    ),

                                away_team:
                                    item.get(
                                        "away",
                                        "-"
                                    )

                            })


                        if stat_rows:

                            st.dataframe(
                                pd.DataFrame(
                                    stat_rows
                                ),
                                use_container_width=True,
                                hide_index=True
                            )


# ============================================================
# PROSSIMO MODULO
# ============================================================

st.markdown("---")

st.info(
    "🚧 Modulo probabilità in sviluppo: "
    "1X2 • Over/Under • Goal/No Goal"
)
