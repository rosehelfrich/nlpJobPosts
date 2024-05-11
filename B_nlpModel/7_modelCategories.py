from bertopic import BERTopic
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance


df = pd.read_csv('dfPreprocessed.csv', engine='python')
docKey = df['Key Phrases'].to_list()

# Conducted multiple iterations to identify the optimal parameters.
umap_model = UMAP(n_neighbors=12, n_components=6, min_dist=0.0, metric='cosine', random_state=35)
hdbscan_model = HDBSCAN(min_cluster_size=8, prediction_data=True)


model = BERTopic(
        umap_model = umap_model,
        hdbscan_model = hdbscan_model,
        representation_model= MaximalMarginalRelevance(diversity=0.5),
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True),
        )

topics, probs = model.fit_transform(docKey)

tLabels = model.generate_topic_labels(nr_words=4, topic_prefix=False, word_length=None, separator=' ',)
model.set_topic_labels(tLabels)

topicChart = model.get_topic_info().rename(columns={'CustomName': 'Category'}).drop(['Representative_Docs'], axis=1)
topicChart.to_csv("topicChart.csv", index=False)

dfCategorized = model.get_document_info(docKey, df=df).rename(columns={'CustomName': 'Category'}).drop(['Representation', 'Representative_Docs', 'Top_n_words', 'Probability',], axis=1)
dfCategorized.to_csv("dfCategorized.csv", index=False)