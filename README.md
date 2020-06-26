# Text Summariser
Generates summaries from texts - Wikipedia, Textbox and PDF. Uses NLTK for Python to enable tokenisation and core NLP features for Extractive Summarisation, and Hugging Face Transformers for Abstractive Summarisation, with Streamlit for front-end.

## PDF Summariser
Uses Streamlit upload feature, and PDFPlumber to parse text in the PDF. Issues with academic papers which causes some text to become garbled. Works well on non-technical text.

## Wikipedia Summariser
Uses BeautifulSoup to extract text from HTML before passing through the text summarisation engine.

## Textbox Summariser
Basic textbox to allow for copy and paste entry of text for summarisation.


# Extractive vs Abstractive
**Extractive** Summarisation as a technique focusses on determining key themes using frequency analysis of sentences in the corpus of text. **Abstractive** Summarisation uses Transformers to "understand" the key themes at a deeper level and write an entirely new summary, often with newly generated text which does not appear in the corpus itself.

Whilst Abstractive is more effective at generating summaries, due to the large nature of the model, it takes significantly longer to run than Extractive models.

In the demo, you can test out both extractive and abstractive models to compare the difference.

Live demo here: https://summary-generator.herokuapp.com/
