from enum import Enum

class Measure(Enum):
    pearson = 'pearson_correlation'
    jaccard = 'jaccard_similarity'
    necessity = 'necessity_relative_activation'
    sufficiency = 'sufficiency_relative_activation'