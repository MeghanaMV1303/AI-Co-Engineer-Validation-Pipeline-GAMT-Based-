# VARROC AI CO-ENGINEER: PITCH DECK

---

## 💡 Your Selected Problem Statement
**The High Cost of Reactive Manufacturing Validation**
Engineers currently design components in CAD and throw them "over the wall" to manufacturing. Undiscovered geometric flaws (inadequate draft angles, thin walls, sharp stress-concentration edges) pass through tooling, only to fail during injection molding or high-vibration CNC milling. This reactive loop costs millions in trial-outs (T0/T1 failures), scrapped tooling, and delayed time-to-market.

---

## 🚀 Overview of Your Solution
**The GAMT AI Co-Engineer Command Center**
We built a real-time, predictive "Tesla-Dash" AI Co-Engineer. It doesn't just "flag" CAD errors—it simulates them, costs them, and programmatically fixes them. 
Our solution ingests raw STEP geometry, translates it into a Face Adjacency Graph (FAG), and runs a **Geometry-Aware Multi-Task Graph Transformer (GAMT)** to instantly output topological constraints, financial impacts, and an automated Python `cadquery` patch script to resolve the geometry without human intervention.

---

## ⚙️ Process Flow
1. **Geometric Ingestion:** Engineer uploads `assembly.step`.
2. **Graph Translation:** B-Rep logic converts the 3D model into topological nodes & connections.
3. **GAMT Neural Inference:** The Transformer evaluates Draft Angles, Wall Thickness, and Sharp Edges simultaneously.
4. **Hybrid Rule Validation:** Neural intuition merges with strict Hard-Rules pulled live from a Firebase backend.
5. **Timeline Simulation:** The system projects delayed Tooling costs and Production bottlenecks on a vertical timeline.
6. **Auto-Patch Execution:** A verified mathematical fix is deployed, logging a tamper-proof SHA-256 Hash onto the Compliance Vault blockchain.

---

## 🔥 Uniqueness of Your Solution
Our platform pivots from "Static File Checker" to **"Predictive Industrial Intelligence."** Unique features include:
1. **Factory-Sync Simulator:** Dynamically shifts part risk scores depending on the CNC machine selected (e.g., Haas UMC-500 vs Legacy Bridgeport). 
2. **The Live Cost Odometer:** Directly translates topological errors into `₹ Rupees Lost` via scraped tooling penalties.
3. **The Green Ghost:** Evaluates the B-Rep for stress fields and predicts Carbon/Material savings by simulating internal Voronoi latticing.
4. **Blockchain Audit Trail:** Generates cryptographic hashes of any AI-driven geometry shifts to secure Aerospace and Medical legal compliances.

---

## 🔬 Design Details / Working Principle
**Working Principle:**
Instead of point-clouds or meshes, we use **Face Adjacency Graphs (FAGs)**. Faces are nodes; edges are connections. Our custom `TransformerConv` model dynamically learns relationships between perpendicular and parallel faces, identifying topological collisions before tooling begins. 

**Simulations & Benchmarking:**
*   **Legacy Systems (SolidWorks DFM):** Takes 5-10 minutes, requires expert operator, static output.
*   **Varroc AI Co-Engineer:** Takes < 2 seconds, accessible via web dashboard, provides executable auto-fix scripts.

---

## ⚠️ Major Risk And Mitigation
| **Risk to Realize the Concept** | **Mitigation Strategy** |
| :--- | :--- |
| **B-Rep Parsing Failures:** Native STEP parsers often crash on broken or non-manifold geometry. | **Heuristic Fallback:** Our system uses a multi-layered integrity gate that isolates broken topologies and patches them using a synthetic fallback layer before applying Neural weights. |
| **AI Hallucinations (False Positives):** The AI flagging a correct wall as "too thin", resulting in wasted material. | **Hybrid Rules Engine:** GAMT predictions are gated behind absolute deterministic Euclidean math rules stored in Firebase, ensuring physics are always respected. |
| **Hardware Latency:** GNNs on 60,000+ node graphs can bottleneck local engineer laptops. | **Cloud-Compute Payload Limit:** Files > 50MB trigger a simplified heuristic sub-graph parse to prevent RAM overflows. |

---

## 💰 Business Potential & Market Demand
*   **Target Market:** Tier-1 Automotive suppliers (like Varroc), Aerospace engineering, and consumer electronics injection molders.
*   **Market Demand:** The global CAD & PLM market is $15B+. 70% of tooling delays are caused by poor DFM (Design For Manufacturability). 
*   **Business Potential (SaaS Model):** Selling "Seat Licenses" to OEMs. It acts as an instant ROI magnifier. If our *Live Cost Odometer* saves a single $40,000 mold from being scrapped due to an ejection lock (bad draft angle), the software has paid for itself for a decade.

---

## 📅 Plan And Budget Needed
**Milestones (6-Month Realization):**
*   **Month 1-2 (Alpha):** Expand GAMT dataset from 10,000 to 500,000 synthetic geometries covering complex splines and lofts.
*   **Month 3-4 (Beta):** Native integration into standard PLM tools (Teamcenter) and REST APIs.
*   **Month 5-6 (Pilot):** Live deployment on a Varroc internal production line for shadowing human reviewers.

**Estimated Prototype Budget:**
*   **Cloud Compute (AWS/GCP GPUs) for GNN Training:** ₹ 2,50,000
*   **Enterprise CAD Kernel Licensing (.step/.x_t extractors):** ₹ 3,00,000
*   **Database & Firebase Hosting architecture:** ₹ 50,000
*   **Total MVP Hardware/Compute Budget:** ₹ 6,00,000
