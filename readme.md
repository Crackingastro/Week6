# Intelligent Complaint Analysis

Prepare CFPB complaints for semantic search in 5 steps:

1. **Filter & Clean**  
   - Keep only Credit card, Personal loan, BNPL, Savings, Money transfers  
   - Lowercase, strip boilerplate/special chars, collapse whitespace  
   - Output: `data/filtered_complaints.csv`

2. **Chunk**  
   - Split narratives into 600-char pieces with 100-char overlap

3. **Embed**  
   - GPU-accelerated with `all-MiniLM-L6-v2` → 384-dim vectors

4. **Index**  
   - Store in ChromaDB (`vector_store/`) with `id`, `metadata`, `document`, `embedding`


