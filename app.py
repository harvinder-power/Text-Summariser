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
import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, pipeline

st.sidebar.title("© Harvinder Power")
st.sidebar.header("Summariser Tool")
tool = st.sidebar.selectbox("Tool", ["PDF Summariser", "Wikipedia Summariser", "Textbox Summariser"])
st.sidebar.subheader("Abstractive Summarisation")
st.sidebar.markdown("Abstractive Summarisation implements the BART model which uses Transformers to analyse the whole text and generate a uniquely written summary.")
st.sidebar.markdown("_Note: Abstractive Summarisation is computationally intensive, and can take up to a minute to run the model. Text length is limited to ~800 words due to model constraints._")
abstractive = st.sidebar.checkbox('Use Abstractive Summarisation?', value=True)

##Variable to set maximum size of input to transformer. BART is currently limited to 1024 tokens during summarisation. 
transformerMaxSize = 1024


## ---------- Universal functions ----------##

## ---------- Generate Extractive Summary ----------##
def generateSummary(article_text, lines):
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
    return summary


## ---------- Generate Abstractive Summary ---------- ##
def abstractive_summariser(text):
    summarizer = pipeline(task="summarization", model="t5-small")
    summary = summarizer(text)
    output = summary[0]['summary_text']
    return output


## ---------- Wikipedia Article Summariser ---------- ##
def wikipedia_summariser():
    heading = """
    # Wikipedia Summariser  
    Summary generation from Wikipedia articles. In Extractive Sumarisation mode, generated summaries are based on word frequency to determine important of phrases in the corpus of text.

    Whilst this model can work on other websites, it has mixed results due to variability in CSS styling, leading to variable performance.

    _Inspired by this blog article: https://stackabuse.com/text-summarization-with-nltk-in-python/_
    """
    heading
    user_input = st.text_input("Wikipedia Link:", value="https://en.wikipedia.org/wiki/Machine_learning")
    
    if abstractive == False:
        lines = st.number_input("How many lines for the summary?", value=15)

    if st.button("Summarise"):
        scraped_data = urllib.request.urlopen(user_input)
        article = scraped_data.read()

        parsed_article = bs.BeautifulSoup(article,'lxml')

        paragraphs = parsed_article.find_all('p')

        article_text = ""
        for p in paragraphs:
            article_text += p.text

        if abstractive == True:
            print("using abstractive")
            output = abstractive_summariser(article_text[:transformerMaxSize])
            st.header("Summary")
            st.write(output)
        if abstractive == False:
            print("not using abstractive")
            output = generateSummary(article_text, lines)
            st.header("Summary")
            st.write(output)




## ---------- PDF Summariser ---------- ##
## Helper function to extract words from the PDF ##
def extract_data(feed):
    data = ""
    with pdfplumber.load(feed) as pdf:
        pages = pdf.pages
        for p in pages:
            data = data + p.extract_text()
    return data # build more code to return a dataframe 

## Function to input PDF and run through summary generator ##
def pdf_summariser():
    heading = """
    # PDF Summariser
    Summary generation using PDFPlumber for Python. Works well on most PDF types, but has difficulty with academic papers (better performance in abstractive summarisation mode) due to highly variable layout of text, tables, and images.
    
    _From testing, works best on business based documentation and study notes._
    """
    heading
    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
    if abstractive == False:
        lines = st.number_input("How many lines for the summary?", value=15)

    if uploaded_file is not None:
        df = extract_data(uploaded_file)
        if abstractive == True:
            print("using abstractive")
            output = abstractive_summariser(df[:transformerMaxSize])
        if abstractive == False:
            print("not using abstractive")
            output = generateSummary(df, lines)
        # summary = generateSummary(df, lines)
        st.header("Summary")
        st.write(output)
        



## ---------- Textbox  Summariser ---------- ##
def textbox_summariser():
    heading = """
    # Textbox Summariser  
    Summary generation for free text in Python.

    Text needs be long to ensure the summariser is able to take effect.

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
    
    '''
    user_input = st.text_area("Text:", value=dummy_text)
    if abstractive == False:
        lines = st.number_input("How many lines for the summary?", value=5)
    
    if st.button("Summarise"):
        if abstractive == True:
            print("using abstractive")
            output = abstractive_summariser(user_input[:transformerMaxSize])
            st.header("Summary")
            st.write(output)
        if abstractive == False:
            print("not using abstractive")
            output = generateSummary(user_input, lines)
            st.header("Summary")
            st.write(output)
    
    ## -------------------- OLD ---------------------- ##
    # model = T5ForConditionalGeneration.from_pretrained('t5-small')
    # tokenizer = T5Tokenizer.from_pretrained('t5-small')
    # device = torch.device('cpu')

    # # text ="""
    # # Semi-supervised learning falls between unsupervised learning (without any labeled training data) and supervised learning (with completely labeled training data). A representative book of the machine learning research during the 1960s was the Nilsson's book on Learning Machines, dealing mostly with machine learning for pattern classification. Rule-based machine learning approaches include learning classifier systems, association rule learning, and artificial immune systems. Machine learning algorithms build a mathematical model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so. Machine learning also has intimate ties to optimization: many learning problems are formulated as minimization of some loss function on a training set of examples. Generalization in this context is the ability of a learning machine to perform accurately on new, unseen examples/tasks after having experienced a learning data set. Rule-based machine learning is a general term for any machine learning method that identifies, learns, or evolves "rules" to store, manipulate or apply knowledge. The computational analysis of machine learning algorithms and their performance is a branch of theoretical computer science known as computational learning theory. Unsupervised learning algorithms take a set of data that contains only inputs, and find structure in the data, like grouping or clustering of data points. Performing machine learning involves creating a model, which is trained on some training data and then can process additional data to make predictions. Usually, when training a machine learning model, one needs to collect a large, representative sample of data from a training set. Leo Breiman distinguished two statistical modeling paradigms: data model and algorithmic model, wherein "algorithmic model" means more or less the machine learning algorithms like Random forest. Feature learning is motivated by the fact that machine learning tasks such as classification often require input that is mathematically and computationally convenient to process. Some statisticians have adopted methods from machine learning, leading to a combined field that they call statistical learning. Early classifications for machine learning approaches sometimes divided them into three broad categories, depending on the nature of the "signal" or "feedback" available to the learning system. Sparse dictionary learning is a feature learning method where a training example is represented as a linear combination of basis functions, and is assumed to be a sparse matrix. Association rule learning is a rule-based machine learning method for discovering relationships between variables in large databases. Unsupervised learning can be a goal in itself (discovering hidden patterns in data) or a means towards an end (feature learning). As of 2020, deep learning has become the dominant approach for much ongoing work in the field of machine learning.
    # # """

    # preprocess_text = text.strip().replace("\n","")
    # t5_prepared_Text = "summarize: "+preprocess_text
    # print ("original text preprocessed: \n", preprocess_text)

    # tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # # summmarize 
    # summary_ids = model.generate(tokenized_text,
    #                                     num_beams=4,
    #                                     no_repeat_ngram_size=2,
    #                                     min_length=30,
    #                                     max_length=250,
    #                                     early_stopping=True)

    # output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # st.write(output)


## I miss JS case-switch syntax here :'(
if tool == "Wikipedia Summariser":
    wikipedia_summariser()

if tool == "PDF Summariser":
    pdf_summariser()

if tool == "Textbox Summariser":
    textbox_summariser()

# if tool == "Abstractive Summarisation":
#     abstractive_summariser()