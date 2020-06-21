# Text Summariser
Generates summaries from texts - Wikipedia, Textbox and PDF (WIP). Uses NLTK for Python to enable tokenisation and core NLP features, with Streamlit for front-end.

## PDF Summariser
Uses Streamlit upload feature, and PDFPlumber to parse text in the PDF. Issues with academic papers which causes some text to become garbled. Works well on non-technical text.

## Wikipedia Summariser
Uses BeautifulSoup to extract text from HTML before passing through the text summarisation engine.

## Textbox Summariser
Basic textbox to allow for copy and paste entry of text for summarisation.

Live demo here: https://summary-generator.herokuapp.com/
