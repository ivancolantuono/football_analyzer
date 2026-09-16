import streamlit as st
import subprocess
import sys
import os

st.set_page_config(
    page_title="Football Analyzer",
    page_icon="⚽"
)

st.title("⚽ Football Analyzer")
st.subheader("Test Playwright + SofaScore")


# ============================================================
# INSTALLAZIONE CHROMIUM
# ============================================================

@st.cache_resource
def install_playwright():

    try:

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "playwright",
                "install",
                "chromium"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )

        return (
            result.returncode,
            result.stdout,
            result.stderr
        )

    except Exception as e:

        return (
            -1,
            "",
            str(e)
        )


# ============================================================
# TEST
# ============================================================

if st.button("🚀 Avvia test"):

    st.info(
        "Installazione/controllo Chromium..."
    )

    code, stdout, stderr = install_playwright()

    if code != 0:

        st.error(
            "❌ Installazione Chromium fallita"
        )

        if stdout:
            st.code(stdout)

        if stderr:
            st.code(stderr)

        st.stop()


    st.success(
        "✅ Chromium disponibile"
    )


    # ========================================================
    # PLAYWRIGHT
    # ========================================================

    from playwright.sync_api import sync_playwright

    try:

        with sync_playwright() as p:

            st.info(
                "Avvio Chromium..."
            )

            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu"
                ]
            )

            st.success(
                "✅ Chromium avviato"
            )


            # ------------------------------------------------
            # PAGINA
            # ------------------------------------------------

            page = browser.new_page(
                viewport={
                    "width": 1366,
                    "height": 768
                }
            )


            # ------------------------------------------------
            # SOFASCORE
            # ------------------------------------------------

            url = (
                "https://www.sofascore.com/it/football"
            )

            st.info(
                "Apertura SofaScore..."
            )

            response = page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=60000
            )


            # ------------------------------------------------
            # RISPOSTA
            # ------------------------------------------------

            if response:

                st.write(
                    "Codice HTTP:"
                )

                st.code(
                    str(response.status)
                )


            st.write(
                "URL finale:"
            )

            st.code(
                page.url
            )


            st.write(
                "Titolo:"
            )

            st.code(
                page.title()
            )


            # ------------------------------------------------
            # ATTESA
            # ------------------------------------------------

            page.wait_for_timeout(
                8000
            )


            # ------------------------------------------------
            # TESTO
            # ------------------------------------------------

            text = page.locator(
                "body"
            ).inner_text(
                timeout=20000
            )


            if text:

                st.success(
                    "✅ SofaScore è stato caricato!"
                )

                st.text(
                    text[:15000]
                )

            else:

                st.warning(
                    "Pagina caricata ma nessun testo trovato."
                )


            # ------------------------------------------------
            # SCREENSHOT
            # ------------------------------------------------

            screenshot = page.screenshot(
                full_page=False
            )

            st.image(
                screenshot,
                caption="SofaScore aperto tramite Chromium"
            )


            browser.close()


    except Exception as e:

        st.error(
            "❌ Errore Playwright"
        )

        st.exception(e)
