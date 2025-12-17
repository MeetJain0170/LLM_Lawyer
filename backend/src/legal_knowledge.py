"""
Local Legal Knowledge Base (Smart Keyword Edition)
"""

KNOWLEDGE_BASE = {
    # --- CONSTITUTIONAL LAW ---
    "article 21": {
        "title": "Article 21: Protection of Life and Personal Liberty",
        "content": "No person shall be deprived of his life or personal liberty except according to procedure established by law.\n\n**Significance:** Covers right to live with dignity, privacy, and livelihood.",
        "keywords": ["article 21", "life", "liberty", "privacy", "dignity"]
    },
    
    # --- DOMESTIC VIOLENCE ---
    # We map "beaten", "hit", "home", "abuse" to this entry
    "domestic_violence": {
        "title": "Protection of Women from Domestic Violence Act, 2005",
        "content": "**It is illegal to harm a woman in a domestic relationship.**\n\n**Legal Remedies:**\n1. **Section 12:** File a complaint with a Magistrate.\n2. **Section 18:** Get a Protection Order to stop the abuser from entering the home.\n3. **Section 23:** Get an emergency ex-parte order.\n4. **IPC Section 498A:** Cruelty by husband or relatives is a crime (up to 3 years jail).",
        "keywords": ["domestic violence", "beaten", "beat", "hit", "abuse", "harass", "torture", "wife", "girl", "woman", "home", "husband", "dowry"]
    },

    # --- CRIMINAL LAW ---
    "theft": {
        "title": "Theft (IPC Section 378 / BNS Section 303)",
        "content": "Moving movable property out of possession without consent.\n**Punishment:** Up to 3 years jail (IPC 379).",
        "keywords": ["theft", "steal", "stolen", "robbery", "thief"]
    },
    "murder": {
        "title": "Murder (IPC Section 302 / BNS Section 103)",
        "content": "Intentional killing of a human being.\n**Punishment:** Death sentence or Life Imprisonment.",
        "keywords": ["murder", "kill", "killed", "homicide", "death"]
    },
    "arrest": {
        "title": "Rights of Arrested Person (Article 22 / CrPC 50)",
        "content": "1. Right to know grounds of arrest.\n2. Right to a lawyer.\n3. Right to be presented before Magistrate within 24 hours.",
        "keywords": ["arrest", "police", "custody", "detain", "jail"]
    },
    "fir": {
        "title": "First Information Report (Section 154 CrPC)",
        "content": "Written document prepared by police for cognizable offences. You have a right to a free copy.",
        "keywords": ["fir", "complaint", "police station", "report"]
    },

    # --- PROPERTY ---
    "property": {
        "title": "Transfer of Property Act, 1882",
        "content": "Governs Sale, Mortgage, Lease, and Gift of property.",
        "keywords": ["property", "land", "sell", "buy", "transfer", "flat", "house"]
    },
    "tenant": {
        "title": "Rights of Tenants",
        "content": "A landlord cannot evict without a court order. Tenant has right to essential services (water/electricity).",
        "keywords": ["tenant", "landlord", "rent", "eviction", "evict"]
    },
    
    # --- CYBERCRIME & PRIVACY ---
    "cyber": {
        "title": "Information Technology Act, 2000",
        "content": "The primary law dealing with cybercrime, including hacking, data theft, and online harassment.",
        "keywords": ["cyber", "internet", "online", "computer"]
    },
    "privacy_violation": {
        "title": "Violation of Privacy (Section 66E IT Act) & Obscenity (Section 67A)",
        "content": "**It is a crime to capture or share private/intimate images without consent.**\n\n**Punishment:**\n1. **Section 66E:** Up to 3 years jail for violating privacy.\n2. **Section 67A:** Up to 5 years jail for publishing sexually explicit material electronically.",
        "keywords": ["mms", "leak", "leaked", "nudes", "viral", "video", "photo", "obscene", "porn", "sextortion"]
    },
    "hacking": {
        "title": "Hacking (Section 66 IT Act)",
        "content": "Dishonest use/damage of computer systems.\n**Punishment:** Up to 3 years jail or ₹5 lakh fine.",
        "keywords": ["hack", "hacked", "virus", "malware", "password"]
    }
}

def search_knowledge_base(query):
    query = query.lower()
    
    # Smart Search: Check if ANY keyword exists in the user query
    for topic_key, data in KNOWLEDGE_BASE.items():
        for keyword in data["keywords"]:
            # We look for whole words or phrases
            if keyword in query:
                return data
            
    return None
