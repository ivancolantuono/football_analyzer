import streamlit as st
from playwright.sync_api import sync_playwright


# ============================================================
# CONFIGURAZIONE
# ============================================================

st.set_page_config(
    page_title="Football Analyzer",
    page_icon="⚽",
    layout="wide"
)


# ============================================================
# TITOLO
# ============================================================

st.title("⚽ Football Analyzer")

st.subheader("Test connessione SofaScore")

st.write(
    "Questa versione verifica se Streamlit Cloud "
    "riesce ad aprire SofaScore tramite Chromium."
)


# ============================================================
# URL SOFASCORE
# ============================================================

URL = "https://www.sofascore.com/it/football"


# ============================================================
# PULSANTE
# ============================================================

if st.button("🌐 Avvia test SofaScore"):

    st.info("Avvio Chromium...")

    try:

        # ----------------------------------------------------
        # AVVIO PLAYWRIGHT
        # ----------------------------------------------------

        with sync_playwright() as p:

            browser = p.chromium.launch(
                executable_path="/usr/bin/chromium",
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-setuid-sandbox",
                    "--disable-software-rasterizer"
                ]
            )

            st.success("✅ Chromium avviato correttamente")


            # ------------------------------------------------
            # CREAZIONE PAGINA
            # ------------------------------------------------

            page = browser.new_page(
                viewport={
                    "width": 1366,
                    "height": 768
                },

                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/140.0.0.0 Safari/537.36"
                )
            )


            # ------------------------------------------------
            # APERTURA SOFASCORE
            # ------------------------------------------------

            st.info("Apertura SofaScore...")

            response = page.goto(
                URL,
                wait_until="domcontentloaded",
                timeout=60000
            )


            # ------------------------------------------------
            # RISPOSTA HTTP
            # ------------------------------------------------

            if response:

                st.write(
                    "Codice HTTP ricevuto:"
                )

                st.code(
                    str(response.status)
                )

            else:

                st.warning(
                    "SofaScore non ha restituito "
                    "una risposta HTTP."
                )


            # ------------------------------------------------
            # INFORMAZIONI PAGINA
            # ------------------------------------------------

            st.write("URL finale:")

            st.code(
                page.url
            )


            st.write("Titolo pagina:")

            st.code(
                page.title()
            )


            # ------------------------------------------------
            # ATTESA CARICAMENTO
            # ------------------------------------------------

            st.info(
                "Attendo il caricamento dei dati..."
            )

            page.wait_for_timeout(
                8000
            )


            # ------------------------------------------------
            # LETTURA PAGINA
            # ------------------------------------------------

            try:

                text = page.locator(
                    "body"
                ).inner_text(
                    timeout=15000
                )

            except Exception:

                text = ""


            # ------------------------------------------------
            # RISULTATO
            # ------------------------------------------------

            if text:

                st.success(
                    "✅ Contenuto della pagina recuperato"
                )

                st.write(
                    "Prime informazioni ricevute:"
                )

                st.text(
                    text[:20000]
                )

            else:

                st.warning(
                    "⚠️ La pagina è stata aperta, "
                    "ma non è stato possibile leggere "
                    "il contenuto."
                )


            # ------------------------------------------------
            # SCREENSHOT
            # ------------------------------------------------

            try:

                screenshot = page.screenshot(
                    full_page=False
                )

                st.image(
                    screenshot,
                    caption="SofaScore visto da Chromium"
                )

            except Exception as e:

                st.warning(
                    f"Screenshot non disponibile: {e}"
                )


            # ------------------------------------------------
            # CHIUSURA
            # ------------------------------------------------

            browser.close()


    except Exception as e:

        st.error(
            "❌ Errore durante l'esecuzione di Playwright"
        )

        st.exception(e)
