
# AI Co-Engineer: Intelligent CAD Validation Pipeline (GAMT-Based)

## Overview

AI Co-Engineer is an AI-driven design intelligence system for early-stage CAD validation. It converts CAD geometry into graph representations and uses a Geometry-Aware Multi-Task Graph Transformer (GAMT) to detect manufacturability issues, assess risks, and generate automated fixes.

---

## Problem Statement

In conventional workflows:

* CAD validation is manual and time-consuming
* Errors are detected late in development
* Iterations increase cost and delay

This project enables real-time validation during the design phase.

---

## Proposed Solution

The system:

* Converts CAD models into Face Adjacency Graphs (FAG)
* Uses a Graph Transformer (GAMT) for analysis
* Detects defects and predicts manufacturing risks
* Generates automated CadQuery repair scripts
* Provides real-time visualization

---

## Architecture

```plaintext
CAD (STEP)
   ↓
Face Adjacency Graph (FAG)
   ↓
GAMT Model
   ↓
Validation Engine
   ↓
Auto-Fix Engine (CadQuery)
   ↓
Visualization
```

---

## Core Components

### CAD Processing

Parses STEP files and extracts geometric relationships.

### Graph Construction

Represents faces as nodes and edges as relationships.

### GAMT Model

Performs:

* Defect detection
* Risk scoring
* Localization

### Validation Engine

Identifies:

* Thin walls
* Sharp edges
* Draft issues

### Auto-Fix Engine

Generates rule-guided CadQuery scripts.

### Visualization

Displays defect regions using 3D heatmaps.

---

## Performance

| Metric          | Traditional Systems | AI Co-Engineer |
| --------------- | ------------------- | -------------- |
| Validation Time | 5–10 minutes        | < 3 seconds    |
| Workflow        | Manual              | Automated      |
| Output          | Static              | Interactive    |
| Accuracy        | Expert-dependent    | ~90%           |

---

## Risks and Mitigation

| Risk            | Mitigation            |
| --------------- | --------------------- |
| Parsing failure | Fallback pipeline     |
| False positives | Rule-based validation |
| Data privacy    | On-prem deployment    |

---

## Technology Stack

* PyTorch (Graph Neural Networks)
* OpenCASCADE (CAD processing)
* CadQuery (auto-fix scripts)
* PyVista (visualization)
* Python / Node.js

---

## Project Structure

```plaintext
├── data/                # CAD input files
├── graph/               # FAG generation
├── models/              # GAMT model
├── validation/          # defect detection logic
├── autofix/             # CadQuery scripts
├── visualization/       # heatmap rendering
├── api/                 # backend services
├── frontend/            # UI
├── configs/             # configuration files
├── tests/               # unit and integration tests
└── docs/                # documentation
```

---

## Future Work

* Feedback-driven model improvement
* Integration with industrial CAD tools
* Advanced cost prediction
* Hybrid deployment

---

## Author

Meghana M V
Electronics and Communication Engineering
University of Visvesvaraya College of Engineering

---

## License

Developed for Varroc Eureka 3.0 Challenge.
