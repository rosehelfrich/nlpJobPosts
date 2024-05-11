# Data Science Trends Analysis Project

## Overview

This project is designed to categorize current trends in the data science field by analyzing mid-level job postings related to data analytics, data science, and machine learning. The process involves two main parts: web scraping to collect data, and natural language processing to categorize the information.

## Part I: Data Collection with Web Scraping

**Objective:** Collect and compile data from job postings to create a comprehensive dataset for analysis.

#### A. Zyte_get_url
- **Description:** Retrieves URLs from job postings under "Data + Analytics" for mid-level positions from [BuiltIn](https://builtin.com).
- **Output:** `url_list.json`

#### B. Zyte_get_info
- **Description:** Gathers detailed information from the retrieved URLs, including job title, description, employment type, company, salary, and location.
- **Output:** `raw_info_df.csv` containing 507 unique job posts from January 2024.

**Technology:**
- Beautiful Soup
- Python
- Pandas
- Zyte (formerly known as Scrapy Cloud)

## Part II: Natural Language Processing

**Objective:** Clean and analyze the collected job descriptions to identify prevalent data science technologies and applications.

#### C. 1a_cleanMetadata
- **Description:** Cleans metadata for consistency and enhanced readability.
- **Output:** `metadataCleaned.csv`

#### D. 1b_deepClean
- **Description:** Performs deeper cleaning on job titles and descriptions in preparation for NLP.
- **Output:** `dfCleaned.csv`

#### E. 2_jobSummary
- **Description:** Summarizes each job description using the `facebook/bart-large-cnn` model.
- **Output:** `jobSummaries.pkl`
- **Technology:**  Transformers, BartForConditionalGeneration

#### F. 3_labelPreprocess
- **Description:** Lemmatizes and filters tokens, identifies frequently used words, and creates additional columns for detailed analysis:
  - **Tech Stack**: Derived from cross-referencing `techList.csv` with job descriptions.
  - **Applications**: Originates from a list in the LinkedIn group "Artificial Intelligence, Machine Learning, Data Science & Robotics," updated with model iterations.
  - **Bag of Words**: Compares job descriptions against frequently occurring tokens.
- **Outputs:** `dfPreprocessed.csv`, `visTokenFreq.png`

#### G. 4_modelCategories
- **Description:** Utilizes an unsupervised BERTopic model to process job descriptions based on the "Applications" column and clusters them into relevant topics.
- **Outputs:**
  - `dfCategorized.csv`: Jobs assigned a topic or labeled "general" if insufficient data.
  - `visCategory.png`: Visualization of data science topics.
  - Additional visualizations based on the model performance.
- **Technology:** BERTopic


## Technology

This project utilizes the following programming languages and libraries:

- **DataMapPlot**: 0.2.2
- **Matplotlib**: 3.8
- **NLTK**: 3.8.1 
- **NumPy**: 1.26
- **Pandas**: 1.5.3
- **Python**: 3.11
- **PyTorch**: 2.2.1
- **Scikit-learn (Sklearn)**: 1.2.2
- **SciPy**: 1.11
- **Seaborn**: 0.12.2
- **SentenceTransformers**: 2.3.1
- **Spacy**: 3.7.2
- **Transformers**: 4.36.2

