Ce projet explore une approche innovante combinant algorithmes classiques de découverte de dépendances fonctionnelles (FD) et modèles de langage (LLM) pour identifier des relations valides et sémantiquement pertinentes dans les bases de données. Notre méthodologie hybride vise à distinguer les FD significatives des FD accidentelles, en exploitant les forces complémentaires de l'analyse algorithmique et du raisonnement sémantique.

## Contexte
Les dépendances fonctionnelles sont des contraintes fondamentales dans les bases de données relationnelles. Leur découverte automatique est essentielle pour :

Améliorer la qualité des données

Détecter des anomalies

Normaliser les schémas de bases de données

Comprendre la sémantique sous-jacente des données

## Objectifs
### Objectif Principal
- Développer une méthodologie hybride permettant de :

- Analyser les FD découvertes par des algorithmes classiques (TANE, FastFD)

- Évaluer leur pertinence sémantique avec l'aide de LLM

- Distinguer les FD significatives des FD accidentelles

- Concevoir un pipeline intégrant analyse algorithmique et raisonnement assisté

### Objectifs Spécifiques
- Implémenter un système de filtrage sémantique des FD

- Développer une interface interactive pour l'exploration des résultats

- Évaluer l'efficacité de l'approche hybride sur différents datasets

- Documenter les avantages et limitations de la méthode

## Organisation du Projet
Le projet est structuré en plusieurs tâches interconnectées :

### 1. Task Set 1: Interprétation des FD Algorithmiques
- Analyse des FD minimales découvertes par l'algorithme TANE

- Extraction et normalisation des FD depuis différents formats

- Identification des FD triviales et suspectes

- Statistiques descriptives sur les patterns découverts

### 2. Task Set 2: Découverte Sémantique Assistée par LLM
- Évaluation de la plausibilité sémantique des FD

- Catégorisation des FD (meaningful, accidental, encoding-based, etc.)

- Comparaison entre jugement LLM et validation humaine

- Analyse des désaccords et des limites du LLM

### 3. Task Set 3: Échantillonnage et Hypothèses de FD
- Stratégies d'échantillonnage pour la génération d'hypothèses

- Validation des FD suggérées par le LLM

- Analyse des faux positifs et des FD non minimales

- Insights sur la généralisation à partir d'échantillons

### 4. Task Set 4: Système Hybride de Découverte de FD
- Conception d'un pipeline intégrant algorithmes et LLM

- Architecture modulaire avec filtrage en plusieurs étapes

- Implémentation en Python avec Streamlit

- Évaluation quantitative et qualitative du système

### 5. Interface Utilisateur
- Développement d'une interface web interactive avec Streamlit

- Support des datasets prédéfinis et fichiers CSV personnalisés

- Visualisation interactive des FD découvertes

- Export multi-format des résultats
