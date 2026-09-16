import streamlit as st
from playwright.sync_api import sync_playwright

st.set_page_config(
    page_title="Football Analyzer",
    page_icon="⚽"
)

st.title("⚽ Test SofaScore - Browser")

url = "https://www.sofascore.com/it/football"

if st.button("🌐 Apri SofaScore"):

    with st.spinner("Apertura di SofaScore..."):

        try:

            with sync_playwright() as p:

                browser = p.chromium.launch(
                    headless=True
                )

                page = browser.new_page(
                    viewport={
                        "width": 1366,
                        "height": 768
                    }
                )

                page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=60000
                )

                st.success(
                    "✅ SofaScore è stato aperto"
                )

                st.write(
                    "Titolo pagina:"
                )

                st.code(
                    page.title()
                )

                st.write(
                    "URL:"
                )

                st.code(
                    page.url
                )

                # Aspettiamo il caricamento
                page.wait_for_timeout(5000)

                # Testo della pagina
                text = page.locator(
                    "body"
                ).inner_text()

                st.write(
                    "Testo recuperato dalla pagina:"
                )

                st.text(
                    text[:10000]
                )

                browser.close()

        except Exception as e:

            st.error(
                "❌ Errore Playwright"
            )

            st.exception(e)
