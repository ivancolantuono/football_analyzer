import streamlit as st
import requests
from datetime import datetime

st.set_page_config(
    page_title="Football Analyzer",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football Analyzer")
st.caption("Analisi partite e statistiche")

st.sidebar.header("Controlli")

data = st.sidebar.date_input(
    "Data",
    datetime.now().date()
)

if st.sidebar.button("🔄 Aggiorna partite"):

    url = (
        "https://www.sofascore.com/api/v1/"
        f"sport/football/scheduled-events/{data}"
    )

    try:

        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
            timeout=15
        )

        if response.status_code != 200:
            st.error(
                f"Errore Sofascore: "
                f"{response.status_code}"
            )
            st.stop()

        dati = response.json()

        eventi = dati.get("events", [])

        st.success(
            f"Trovate {len(eventi)} partite"
        )

        partite = []

        for evento in eventi:

            casa = evento.get(
                "homeTeam", {}
            ).get("name", "?")

            trasferta = evento.get(
                "awayTeam", {}
            ).get("name", "?")

            campionato = evento.get(
                "tournament", {}
            ).get("name", "?")

            timestamp = evento.get(
                "startTimestamp"
            )

            if timestamp:
                ora = datetime.fromtimestamp(
                    timestamp
                ).strftime("%H:%M")
            else:
                ora = "?"

            partite.append({
                "Ora": ora,
                "Campionato": campionato,
                "Casa": casa,
                "Trasferta": trasferta
            })

        if partite:

            st.dataframe(
                partite,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.warning(
                "Nessuna partita trovata."
            )

    except Exception as e:

        st.error(
            f"Errore: {e}"
        )

else:

    st.info(
        "Seleziona la data e premi "
        "'Aggiorna partite'."
    )
