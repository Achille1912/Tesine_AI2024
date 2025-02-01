# Infrastruttura di Comunicazione Wireless

## Contesto del Problema

Una startup di telecomunicazioni deve progettare un'infrastruttura di rete wireless per connettere 20 siti strategici in una regione montuosa, minimizzando i costi di installazione e massimizzando l'efficienza della connessione.

## Specifiche del Problema

- Rete composta da 20 nodi di comunicazione
- Ogni nodo rappresenta un sito potenziale per stazioni ripetitori
- I costi di connessione variano in base a:
    - Distanza geografica
    - Difficoltà di terreno
    - Investimento infrastrutturale

## Obiettivi

1. Implementare l'algoritmo per ottenere il **Minimum Spanning Tree (MST)**
2. Utilizzare **Kruskal's algorithm** o **Prim's algorithm**
3. Ottimizzare la connessione minimizzando:
    - Costi di installazione
    - Lunghezza totale dei cavi/connessioni
    - Impatto ambientale

## Vincoli

- Tutti i 20 nodi devono essere connessi
- Considerare le caratteristiche del territorio
- Gestire differenti tipologie di costi di connessione

## Fasi del Progetto

### Fase 1: Sviluppo dell'Algoritmo

- Implementare MST con criteri personalizzati
- Considerare:
    - Costi di connessione
    - Qualità del segnale
    - Difficoltà di installazione

### Fase 2: Simulazione degli Scenari

1. **Scenario Montagna**
    - Obiettivo: Connettere siti in zona alpina
    - Vincoli: Minimizzare lunghezza cavi
    - Considerare dislivelli e difficoltà territoriali
2. **Scenario Zona Sismica**
    - Obiettivo: Garantire ridondanza di comunicazione
    - Vincoli: Massimizzare affidabilità connessioni
    - Considerare vulnerabilità del territorio
3. **Scenario Ottimizzazione Energetica**
    - Obiettivo: Minimizzare consumo energetico
    - Vincoli: Limitare numero di ripetitori
    - Valutare efficienza energetica dei siti

## Output Richiesti per Ogni Scenario

- Albero di connessione minimo
- Costo totale dell'infrastruttura
- Analisi dei nodi critici
- Confronto tra diversi algoritmi MST
- Visualizzazione grafica della rete

## Esecuzione del main.py

```
python main.py --scenario mountain

python main.py --scenario seismic

python main.py --scenario energy

python main.py --help
```