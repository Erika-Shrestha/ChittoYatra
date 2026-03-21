# ChittoYatra_23048598_ErikaShrestha

<img width="943" height="497" alt="Picture2" src="https://github.com/user-attachments/assets/f622f454-ee43-4d2e-851f-c7348d8ce02d" />

**ChittoYatra** is an unsupervised machine learning system that optimizes **last-mile delivery** in Nepal's growing Q-Commerce sector. By clustering GPS-based delivery orders, the system enables smarter rider allocation and bulk order batching — reducing delivery times and operational costs.
 
Q-Commerce in Nepal is growing at a CAGR of 25%, with key players like Foodmandu, Daraz, and Bigmart driving demand. ChittoYatra addresses inefficiencies in rider allocation that cost businesses up to 53% in operational risk.

Three clustering algorithms are compared:
 
| Algorithm | Clusters Found | Best Use Case |
|-----------|---------------|---------------|
| **K-Means** | 39 | Daily rider zone allocation — every order assigned |
| **DBSCAN** | 35 + 421 noise | Identifying high-demand zones, filtering GPS errors |
| **Hierarchical** | 15 | Strategic hub planning & mini-warehouse placement |

## Dataset
 
- **Source:** [LaDe-D Dataset](https://huggingface.co/datasets) — Hugging Face
- **Total Documents:** 931,351 real-world delivery records
- **Key Features:** GPS coordinates (lat/lng), accept time, delivery time, courier ID
- **File Format:** Parquet → CSV
- **After filtering:** Orders ≤ 120 min delivery duration (Q-Commerce compatible)
 
## Results
 
| Metric | K-Means (k=39) | DBSCAN (ε=0.15) | Hierarchical (h=20) |
|--------|---------------|-----------------|---------------------|
| Silhouette Score | **0.653** | 0.61 | 0.58 |
| Noise Handling | Forces all points into clusters | Filters 421 noise points | Groups noise into nearest branch |
| Key Parameter | WCSS / Elbow | Epsilon | Dendrogram height |
 
**K-Means** achieved the highest Silhouette Score of **0.653** at k=39, making it the best performer for daily operational use.
