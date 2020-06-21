import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import bs4 as bs
import urllib.request
import re
import nltk
import heapq
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import io
import pdfplumber

st.sidebar.title("© Harvinder Power")
st.sidebar.header("Summarise")
tool = st.sidebar.selectbox("Tool", ["Wikipedia Summariser", "Textbox Summariser", "PDF Summariser"])


######### Wikipedia Article Summariser #########
def wikipedia_summariser():
    heading = """
    # Wikipedia Summariser  
    _Summary generation using NLTK for Python. Generated summaries are based on word frequency to determine important of phrases in the corpus of text. For better results on domain-specific text, a trained model should be used._

    _Whilst this model can work on other websites, it has mixed results due to variability in CSS styling, leading to poor results on some websites._

    _Inspired by this blog article: https://stackabuse.com/text-summarization-with-nltk-in-python/_
    """
    heading
    user_input = st.text_input("Wikipedia Link:", value="https://en.wikipedia.org/wiki/Machine_learning")
    lines = st.number_input("How many lines for the summary?", value=15)

    scraped_data = urllib.request.urlopen(user_input)
    article = scraped_data.read()

    parsed_article = bs.BeautifulSoup(article,'lxml')

    paragraphs = parsed_article.find_all('p')

    article_text = ""

    for p in paragraphs:
        article_text += p.text

    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
    sentence_list = nltk.sent_tokenize(article_text)
    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(lines, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)

    summary



######### Wikipedia Article Summariser #########



def pdf_summariser():
    heading = """
    # PDF Summariser  (Coming Soon)
    _Summary generation using PDFPlumber and NLTK for Python. Coming soon._
    """
    heading
















######### Textbox  Summariser #########



def textbox_summariser():
    heading = """
    # Textbox Summariser  
    _Summary generation for free text in Python using NLTK._

    _Text needs be long to ensure the summariser is able to take effect._

    Example text from this article: https://www.bbc.co.uk/news/science-environment-53119686
    """
    heading
    dummy_text = '''
    We've just become a little less ignorant about Planet Earth.
    The initiative that seeks to galvanise the creation of a full map of the ocean floor says one-fifth of this task has now been completed.
    When the Nippon Foundation-GEBCO Seabed 2030 Project was launched in 2017, only 6% of the global ocean bottom had been surveyed to what might be called modern standards.
    That number now stands at 19%, up from 15% in just the last year.
    Some 14.5 million sq km of new bathymetric (depth) data was included in the GEBCO grid in 2019 - an area equivalent to almost twice that of Australia.
    It does, however, still leave a great swathe of the planet in need of mapping to an acceptable degree.
    "Today we stand at the 19% level. That means we've got another 81% of the oceans still to survey, still to map. That's an area about twice the size of Mars that we have to capture in the next decade," project director Jamie McMichael-Phillips told BBC News.

    The map at the top of this page illustrates the challenge faced by GEBCO in the coming years.
    Black represents those areas where we have yet to get direct echosounding measurements of the shape of the ocean floor. Blues correspond to water depth (deeper is purple, shallower is lighter blue).
    It's not true to say we have no idea of what's in the black zones; satellites have actually taught us a great deal. Certain spacecraft carry altimeter instruments that can infer seafloor topography from the way its gravity sculpts the water surface above - but this only gives a best resolution at over a kilometre, and Seabed 2030 has a desire for a resolution of at least 100m everywhere.

    Better seafloor maps are needed for a host of reasons.
    They are essential for navigation, of course, and for laying underwater cables and pipelines.
    They are also important for fisheries management and conservation, because it is around the underwater mountains that wildlife tends to congregate. Each seamount is a biodiversity hotspot.
    In addition, the rugged seafloor influences the behaviour of ocean currents and the vertical mixing of water.
    This is information required to improve the models that forecast future climate change - because it is the oceans that play a critical role in moving heat around the planet. And if you want to understand precisely how sea-levels will rise in different parts of the world, good ocean-floor maps are a must.
    Much of the data that's been imported into the GEBCO grid recently has been in existence for some time but was "sitting on a shelf" out of the public domain. The companies, institutions and governments that were holding this information have now handed it over - and there is probably a lot more of this hidden resource still to be released.

    But new acquisitions will also be required. Some of these will come from a great crowdsourcing effort - from ships, big and small, routinely operating their echo-sounding equipment as they transit the globe. Even small vessels - fishing boats and yachts - can play their part by attaching data-loggers to their sonar and navigation equipment.
    One very effective strategy is evidenced by the British Antarctic Survey (BAS), which operates in the more remote parts of the globe - and that is simply to mix up the routes taken by ships.
    "Very early on we adopted the ethos that data should be collected on passage - on the way to where we were going, not just at the site of interest," explained BAS scientist Dr Rob Larter.
    "A beautiful example of this is the recent bathymetric map of the Drake Passage area (between South America and Antarctica). A lot of that was acquired by different research projects as they fanned out and moved back and forth to the places they were going."

    New technology will be absolutely central to the GEBCO quest.
    Ocean Infinity, a prominent UK-US company that conducts seafloor surveys, is currently building a fleet of robotic surface vessels through a subsidiary it calls Armada. This start-up's MD, Dan Hook, says low-cost, uncrewed vehicles may be the only way to close some of the gaps in the more out-of-the-way locations in the 2030 grid.
    He told BBC News: "When you look at the the mapping of the seabed in areas closer to shore, you see the business case very quickly. Whether it's for wind farms or cable-laying - there are lots of people that want to know what's down there. But when it's those very remote areas of the planet, the case then is really only a scientific one."
    Jamie McMichael-Phillips is confident his project's target can be met if everyone pulls together.
    "I am confident, but to do it we will need partnerships. We need governments, we need industry, we need academics, we need philanthropists, and we need citizen scientists. We need all these individuals to come together if we're to deliver an ocean map that is absolutely fundamental and essential to humankind."
    GEBCO stands for General Bathymetric Chart of the Oceans. It is the only intergovernmental organisation with a mandate to map the entire ocean floor. The latest status of its Seabed 2030 project was announced to coincide with World Hydrography Day.
    '''
    user_input = st.text_area("Text:", value=dummy_text)
    if st.button("Summarise"):
        output = run_summarization(user_input)
    


def _create_frequency_table(text_string) -> dict:
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _score_sentences(sentences, freqTable) -> dict:


    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        word_count_in_sentence_except_stop_words = 0
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                word_count_in_sentence_except_stop_words += 1
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        if sentence[:10] in sentenceValue:
            sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] / word_count_in_sentence_except_stop_words


    return sentenceValue


def _find_average_score(sentenceValue) -> int:

    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original text
    average = (sumValues / len(sentenceValue))

    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


def run_summarization(text):
    # 1 Create the word frequency table
    freq_table = _create_frequency_table(text)


    # 2 Tokenize the sentences
    sentences = sent_tokenize(text)

    # 3 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(sentences, freq_table)

    # 4 Find the threshold
    threshold = _find_average_score(sentence_scores)

    # 5 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, 1.3 * threshold)

    summary




























## I miss JS case-switch syntax here :'(
if tool == "Wikipedia Summariser":
    wikipedia_summariser()

if tool == "PDF Summariser":
    pdf_summariser()

if tool == "Textbox Summariser":
    textbox_summariser()