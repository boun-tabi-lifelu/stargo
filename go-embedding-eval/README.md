# OWL2Vec* Evaluation on Gene Ontology (GO)

This folder contains evaluation code and results comparing OWL2Vec* against other deep learning-based GO embedding techniques, Anc2Vec, GT2Vec, BioBERT, using the latest Gene Ontology version (2024-09-08).

## Dataset
- Gene Ontology (GO) version: 2024-09-08
- Format: OWL
- Source: [Gene Ontology Downloads](http://geneontology.org/docs/download-ontology/)

## Embedding Methods Compared
1. OWL2Vec*
   - Uses both structural features and lexical information from GO
   - Combines RDF graph walks with word embeddings
   - Parameters used match those in the original OWL2Vec* paper

2. Anc2Vec
   - Ancestor-based embedding approach
   - Focuses on hierarchical relationships in GO
   - Implementation based on original Anc2Vec paper

## Evaluation Tasks
1. Subsumption Prediction
   - Predicting hierarchical (is-a) relationships between GO terms
   - Split: 70% training, 10% validation, 20% testing
   - Metrics: Precision, Recall, F1-score

2. Semantic Similarity
   - Computing similarity between GO term pairs
   - Comparison against established GO semantic similarity measures
   - Correlation analysis with manual annotations


