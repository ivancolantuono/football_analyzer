import streamlit as st
from curl_cffi import requests

st.set_page_config(
    page_title="Test SofaScore",
    page_icon="⚽"
)

st.title("⚽ Test connessione SofaScore")

URL = "https://api.sofascore.com/api/v1/sport/football/scheduled-events/2026-09-16"

st.write("URL utilizzato:")
st.code(URL)

try:

    session = requests.Session(
        impersonate="chrome"
    )

    response = session.get(
        URL,
        timeout=30
    )

    st.write("Codice HTTP:")
    st.code(response.status_code)

    st.write("Headers ricevuti:")
    st.json(dict(response.headers))

    if response.status_code == 200:

        data = response.json()

        st.success("✅ CONNESSIONE SOFASCORE RIUSCITA")

        st.write(
            "Numero partite:",
            len(data.get("events", []))
        )

        st.json(
            data
        )

    else:

        st.error(
            f"❌ SofaScore ha restituito HTTP {response.status_code}"
        )

        st.write(
            "Risposta del server:"
        )

        st.code(
            response.text[:5000]
        )

except Exception as e:

    st.error(
        "❌ ERRORE PYTHON"
    )

    st.exception(e)
