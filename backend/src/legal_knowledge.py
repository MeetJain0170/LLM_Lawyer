"""
Local Legal Knowledge Base
Acts as a Ground Truth safety net for the AI.
"""

KNOWLEDGE_BASE = {
    # --- CRIMINAL LAW (IPC/BNS) ---
    "theft": {
        "title": "Theft (IPC Section 378 / BNS Section 303)",
        "content": "Theft is defined as moving movable property out of the possession of any person without that person's consent. \n\n**Punishment:** Up to 3 years imprisonment, fine, or both (IPC Section 379)."
    },
    "murder": {
        "title": "Murder (IPC Section 302 / BNS Section 103)",
        "content": "Murder is the intentional killing of a human being. \n\n**Punishment:** Death sentence or imprisonment for life, and fine."
    },
    "arrest": {
        "title": "Rights of Arrested Person (Article 22)",
        "content": "1. Right to be informed of grounds of arrest.\n2. Right to consult a lawyer.\n3. Right to be produced before a magistrate within 24 hours."
    },
    
    # --- DOMESTIC VIOLENCE ---
    "domestic violence": {
        "title": "Domestic Violence Act, 2005",
        "content": "Protects women from physical, emotional, sexual, and economic abuse by any person in a domestic relationship. \n\n**Remedy:** You can file a DIR (Domestic Incident Report) with a Protection Officer or Magistrate."
    },
    "dowry": {
        "title": "Dowry Death (IPC 304B)",
        "content": "If a woman dies within 7 years of marriage due to burns or bodily injury and it is shown she was harassed for dowry, it is Dowry Death.\n\n**Punishment:** Minimum 7 years imprisonment, up to life."
    },

    # --- PROPERTY ---
    "property": {
        "title": "Transfer of Property Act",
        "content": "Governs the transfer of property by act of parties. Sale, Mortgage, Lease, Exchange, and Gift are key modes of transfer."
    },
    "tenant": {
        "title": "Rights of Tenants",
        "content": "A landlord cannot evict a tenant without a valid court order. The tenant has the right to peaceful possession and essential services (water/electricity)."
    },

    # --- CYBERCRIME ---
    "cyber": {
        "title": "IT Act, 2000",
        "content": "The primary law in India dealing with cybercrime and electronic commerce."
    },
    "hacking": {
        "title": "Hacking (Section 66 IT Act)",
        "content": "Any person who dishonestly or fraudulently does any act referred to in Section 43 (damage to computer system). \n\n**Punishment:** Up to 3 years imprisonment or ₹5 lakh fine."
    },
    "identity theft": {
        "title": "Identity Theft (Section 66C IT Act)",
        "content": "Fraudulently making use of the electronic signature, password, or any other unique identification feature of any other person.\n\n**Punishment:** Up to 3 years imprisonment."
    }
}

def search_knowledge_base(query):
    query = query.lower()
    best_match = None
    
    # Simple keyword matching
    for key in KNOWLEDGE_BASE:
        if key in query:
            return KNOWLEDGE_BASE[key]
            
    return None