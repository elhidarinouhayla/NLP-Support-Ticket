# NLP-Support-Ticket


##  Vue d'ensemble

Système automatisé de classification de tickets support IT utilisant le traitement du langage naturel (NLP) et des pratiques MLOps pour industrialiser le processus de catégorisation des emails clients.

##  Objectifs

- **Automatiser** la classification des tickets support
- **Améliorer** le routage et la priorisation des demandes
- **Garantir** la stabilité du modèle face à l'évolution du vocabulaire client
- **Superviser** la performance et la santé de l'infrastructure

##  Architecture Complète du Projet

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         EMAILS SUPPORT CLIENT                                   │      
│                       (Subject + Body + Metadata)                               │     
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                       PIPELINE ML BATCH (Python)                               │
│                                                                                │
│  ┌───────────────────────────────────────────────────────────────────────┐     │
│  │  ÉTAPE 1: PRÉTRAITEMENT NLP                                           │     │
│  │                                                                       │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐│     │ 
│  │  │ Normalisation│→ │  Suppression │→ │ Tokenisation │→ │  Stopwords  ││     │
│  │  │  (lowercase) │  │  Ponctuation │  │              │  │   Removal   ││     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘│     │ 
│  └─────────────────────────────────┬─────────────────────────────────────┘     │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  ÉTAPE 2:  GÉNÉRATION EMBEDDINGS                                        │   │
│  │                                                                         │   │
│  │  ┌──────────────────────┐      ┌──────────────────┐                     │   │
│  │  │  Hugging Face Model  │  →   │   Encodage       │                     │   │
│  │  │ sentence-transformers│      │  Normalisation   │                     │   │
│  │  └──────────────────────┘      └────────┬─────────┘                     │   │
│  │                                          │                              │   │
│  │                                          ▼                              │   │
│  │                              ┌────────────────────────┐                 │   │
│  │                              │    ChromaDB            │                 │   │
│  │                              │   Base Vectorielle     │                 │   │
│  │                              │   - Embeddings         │                 │   │
│  │                              │   - Metadata           │                 │   │
│  │                              │   - Indexation         │                 │   │
│  │                              └────────────────────────┘                 │   │
│  └─────────────────────────────────┬───────────────────────────────────────┘   │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  ÉTAPE 3:  CLASSIFICATION ML                                            │   │
│  │                                                                         │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │   │
│  │  │  Train/Test  │→ │ scikit-learn │→ │  Évaluation  │→ │ Sauvegarde  │  │   │
│  │  │    Split     │  │ Entraînement │  │  Métriques   │  │   Modèle    │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │   │
│  └─────────────────────────────────┬───────────────────────────────────────┘   │
└────────────────────────────────────┼───────────────────────────────────────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                ▼                    ▼                    ▼
┌──────────────────────────┐ ┌──────────────────┐ ┌─────────────────────────┐
│   MONITORING ML          │ │   CONTENEURS     │ │   CI/CD                 │
│  (Evidently AI)          │ │  & ORCHESTRATION │ │  (GitHub Actions)       │
│                          │ │                  │ │                         │
│  ┌────────────────────┐  │ │  ┌────────────┐  │ │  ┌────────────────────┐ │
│  │  Data Drift        │  │ │  │  Docker    │  │ │  │  1. Lint & Test    │ │
│  │  Detection         │  │ │  │  Image     │  │ │  │                    │ │
│  └────────────────────┘  │ │  └─────┬──────┘  │ │  └─────────┬──────────┘ │
│                          │ │        │         │ │            │            │
│  ┌────────────────────┐  │ │        ▼         │ │            ▼            │
│  │  Prediction Drift  │  │ │  ┌────────────┐  │ │  ┌────────────────────┐ │
│  │  Monitoring        │  │ │  │  Docker    │  │ │  │  2. Build Docker   │ │
│  └────────────────────┘  │ │  │  Compose   │  │ │  │                    │ │
│                          │ │  └─────┬──────┘  │ │  └─────────┬──────────┘ │
│  ┌────────────────────┐  │ │        │         │ │            │            │
│  │  Rapports HTML     │  │ │        ▼         │ │            ▼            │
│  │  Interactifs       │  │ │  ┌────────────┐  │ │  ┌────────────────────┐ │
│  └────────────────────┘  │ │  │ Kubernetes │  │ │  │  3. Deploy K8s     │ │
│                          │ │  │  (minikube)│  │ │  │                    │ │
│                          │ │  │ Job/CronJob│  │ │  └────────────────────┘ │
└──────────────────────────┘ └────────┬─────────┘ └─────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   MONITORING INFRASTRUCTURE                                     │
│                      (Prometheus + Grafana)                                     │
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │
│  │     Prometheus  │    │   Grafana       │    │    cAdvisor     │              │
│  │   Port: 9090    │ ←→ │   Port: 3000    │ ←→ │   Port: 8080    │              │
│  │                 │    │                 │    │                 │              │
│  │  - Collecte     │    │  - Dashboards   │    │  - Métriques    │              │
│  │  - Alerting     │    │  - Visualisation│    │    Containers   │              │
│  │  - Time Series  │    │  - Alertes      │    │  - Docker Stats │              │
│  └────────┬────────┘    └─────────────────┘    └─────────────────┘              │
│           │                                                                     │
│           │             ┌─────────────────┐                                     │
│           └───────────→ │    Node Exporter│                                     │
│                         │   Port: 9100    │                                     │
│                         │                 │                                     │
│                         │  - CPU/RAM      │                                     │
│                         │  - Disque       │                                     │
│                         │  - Réseau       │                                     │
│                         └─────────────────┘                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        RÉSULTATS & SORTIES                                      │
│                                                                                 │
│     Tickets Classifiés  │   Rapports Drift  │   Métriques  │   Prédictions      │
└─────────────────────────────────────────────────────────────────────────────────┘
```






##  Structure du Projet

```
.
├── data/                  
├── notebooks/              
├── src/
│   ├── preprocessing/      
│   ├── embeddings/         
│   ├── training/           
│   └── monitoring/         
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── k8s/                    
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
├── .github/workflows/      
└── README.md
```



### Installation

```bash
# Cloner le repository
git clone https://github.com/elhidarinouhayla/NLP-Support-Ticket.git
cd  NLP-Support-Ticket


# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```



##  Étapes du Pipeline

###  Analyse Exploratoire & Préparation NLP

- Distribution des types de tickets
- Analyse de la longueur des emails
- Fusion des champs texte (subject + body)
- Nettoyage NLP:
  - Normalisation (lowercase)
  - Suppression ponctuation
  - Tokenisation
  - Suppression des stopwords

###  Génération d'Embeddings

- Sélection d'un modèle pré-entraîné Hugging Face
- Encodage des textes en vecteurs sémantiques
- Normalisation des embeddings
- Indexation dans ChromaDB

###  Entraînement du Modèle

- Séparation train/test
- Entraînement avec scikit-learn
- Évaluation des performances (accuracy, precision, recall, F1)



###  Conteneurisation & Orchestration

- Dockerisation du pipeline
- Déploiement sur Kubernetes (Job/CronJob)
- CI/CD avec GitHub Actions

###  Monitoring Infrastructure

**Services déployés:**

```yaml
- Prometheus (port 9090)  # Collecte des métriques
- Grafana (port 3000)     # Visualisation
- cAdvisor (port 8080)    # Métriques containers
- Node Exporter (port 9100) # Métriques système
```

**Lancer le monitoring:**

```bash
cd monitoring/
docker-compose up -d
```

**Accès:**
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

##  CI/CD Pipeline

Le workflow GitHub Actions s'exécute automatiquement sur les branches `main` et `develop`:

1. **Lint**: Vérification de la qualité du code
2. **Build**: Construction de l'image Docker




##  Configuration

### Prometheus (prometheus.yml)

```yaml
global:
  scrape_interval: 5s 

scrape_configs:
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
  
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```
