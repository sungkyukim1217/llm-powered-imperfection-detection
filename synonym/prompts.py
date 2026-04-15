SYSTEM_PROMPT_SYNONYMOUS_STEP1 = """
You are an expert Process Mining Data Pre-processor.
Your goal is to filter a raw list of activity names based on specific criteria provided in the User Prompt.

### KNOWLEDGE BASE: IMPERFECTION PATTERNS
Use these definitions to identify which labels belong to which category.

1.  **Polluted Labels (Mutable Qualifiers):**
    Labels that share a immutable boiler-plate text but differ due to mutable text (e.g., embedded IDs or codes).
    * **Detection Criteria:**
        * **Long Numeric IDs:** 8+ digits (e.g., `20260122`, `9988776655`).
        * **Mixed Codes:** 6+ alphanumeric chars (e.g., `XJ9281`, `Ref_A1B2C3`).
        * **Delimiters:** Attached via `_`, `-`, `:`, `/`, `#`, `.`, or space.

2.  **Distorted Labels (Character-Level Corruption):**
    Labels containing specific character-level corruptions (typos, OCR faults) of a canonical form. Unlike synonyms, these are "Noise".
    * **Detection Criteria:**
        1.  **Case Mutation:** Identical spelling, different capitalization (e.g., "Open" vs "open" vs "OPEN").
        2.  **Character Omission:** Exactly ONE missing character (e.g., "Invoce" vs "Invoice").
        3.  **Character Insertion:** Exactly ONE extra character (e.g., "Innvoice" vs "Invoice").
        4.  **Character Transposition:** Two adjacent characters swapped (e.g., "Ivnoice" vs "Invoice").
        5.  **Keyboard Proximity:** Exactly ONE character substituted by a QWERTY neighbor (e.g., "Invoicr" vs "Invoice").

3. **Synonymous Labels (Semantic Equivalence):**
   Labels that are syntactically different (often substantially) but share the same semantic meaning and represent the exact same business process step. 
   * **Detection Criteria (Ontology Rules):**
        1. **Linguistic & Domain Synonyms:** Different words representing the same concept within the process context (e.g., "Ship Item" vs "Dispatch Goods", "DrSeen" vs "Medical Assign").
        2. **Phrase Variation (Verb/Object Shift):** Labels sharing a core component (usually the Object) while using synonymous verbs or adjectives (e.g., "Create Invoice" vs "Generate Invoice", "Start instance" vs "Start process").
        3. **Grammatical Transformation:** Changing parts of speech (Noun ↔ Verb) or sentence structure while retaining the core meaning (e.g., "Give approval" vs "Approve", "Conduct analysis" vs "Analyze").
        4. **Containment & Refinement:** One label is a concise or verbose version of the other, often omitting non-essential adjectives, prepositions, or 'online/offline' qualifiers (e.g., "Receive signed contract" vs "Receive contract", "Register for course" vs "Register course").

### GLOBAL INSTRUCTION
- **Role:** Function as a logic engine. Do not assume all imperfections exist.
- **Priority:** The strict filtering logic in the **User Prompt** overrides general definitions here.
- **OUTPUT FORMAT:** Always return valid JSON as requested by the User Prompt.

"""

USER_PROMPT_SYNONYMOUS_STEP1 = """
### TASK: Filter Data for 'Synonymous Labels' Candidates

**OBJECTIVE:**
Analyze the provided **INPUT DATA(activity list)** and extract labels related to **Synonymous Labels (Semantic Equivalence)**.
You must **identify and collect** the following components for the output data:
1. All labels that form a **Synonym Group** (two or more labels representing the same process step despite syntactic differences).

**STRICT FILTERING LOGIC:**
1. **Identify Synonym Pairs:** Look for different words or phrases that share the same semantic meaning based on the System Prompt's criteria (Linguistic Synonyms, Phrase Variation, Grammatical Transformation, Containment).
2. **KEEP Related Labels:** If Label A is a semantic synonym of Label B, **KEEP BOTH A and B**. (Unlike Distorted/Polluted, typically all members of a synonym group are valid words, so keep the entire group).
3. **DISCARD Isolated Labels:** If a label is valid but has **NO** semantic synonyms in the provided list, **REMOVE IT**. (e.g., If 'Archive' exists but no synonyms like 'Store' or 'Save' exist, remove 'Archive').
4. **DISCARD Distorted/Polluted:** Remove labels that are purely 'Distorted Labels' (typos) or 'Polluted Labels' (IDs) if they are not part of a 'Synonymous Labels' pattern.

**EDGE CASE HANDLING:**
- If NO Synonymous pairs are found (i.e., all labels are unique/isolated, distorted, or polluted):
    - Return strictly `[]` (with "found": false).
- **Finding NOTHING is a valid result.** Do not force-fit vaguely similar words; strict semantic equivalence is required.

**INPUT DATA:**
{SYNONYM_STEP1_INPUT}

***OUTPUT FORMAT GUIDELINES (PERFORMANCE OPTIMIZED)***
Return a JSON Object with two keys:
1. "found": Boolean (true if synonymous labels exist, false otherwise).
2. "data": List of strings.

**Example (Found):**
{{ "found": true, "data": ["Create Invoice", "Generate Invoice", "Make Bill"] }}

**Example (Not Found - SPEED PRIORITY):**
{{ "found": false, "data": [] }}

**CONSTRAINT:**
- Determine the "found" value FIRST. If false, output `[]` for data immediately.
- Output ONLY the JSON.
"""

SYSTEM_PROMPT_SYNONYM_STEP2  = """
You are an expert Process Mining Analyst.
Your goal is to summarize lists of activity labels into a single, descriptive **Process Stage Name**.

### CORE TASK
You will be given an activity and its lists of **Predecessors** (incoming flow) and **Successors** (outgoing flow).
You must analyze the labels in each list and determine the **Common Business Phase** they represent.

### SUMMARIZATION LOGIC (ABSTRACTION)
1. **Identify the Core Action:** Look at the verbs and objects in the list.
2. **Ignore Noise:** Disregard synonyms, typos, and minor variations.
3. **Formulate a Summary:** Create a short, natural language phrase that encapsulates the collective meaning.

### EXAMPLES (Demonstration Only)
- **Input List:** `["Wrap package", "Box items", "Pack goods", "Containerize"]`
- **Output Summary:** "Packaging Phase"

- **Input List:** `["MRI Scan", "X-Ray taken", "Blood test results"]`
- **Output Summary:** "Medical Diagnosis Stage"

- **Input List:** `["Ticket Resolved", "Issue Fixed", "Close Ticket", "Problem Solved"]`
- **Output Summary:** "Ticket Resolution"

### GLOBAL INSTRUCTION
- **Input:** JSON object with `activity`, `predecessors` (list), and `successors` (list).
- **Output:** JSON object where `predecessors` and `successors` are converted to **Strings** (Summaries).
"""

USER_PROMPT_SYNONYM_STEP2 = """
### TASK: Summarize Contextual Flow Lists

**OBJECTIVE:**
Analyze the **INPUT DATA**. Replace the list of strings in `predecessors` and `successors` with a **Single Summarized String** describing that process stage.

**STRICT EXECUTION STEPS:**
1. **Iterate** through every activity in the input.
2. **Analyze Predecessors:**
   - Read the list of predecessor labels.
   - Abstract their common meaning into one short phrase (e.g., "Quality Check Phase").
   - **Replace** the list with this string.
3. **Analyze Successors:**
   - Read the list of successor labels.
   - Abstract their common meaning into one short phrase.
   - **Replace** the list with this string.

**INPUT DATA:**
{SYNONYM_STEP2_INPUT}

***OUTPUT FORMAT GUIDELINES***
Return a JSON Object with a single key `"summarized_context"`.
The value must be a list of objects where `predecessors` and `successors` are **STRINGS**, not lists.

**Example Output (Mental Model):**
{{
  "summarized_context": [
    {{
      "activity": "Ship Item",
      "predecessors": "Packaging Phase",     // Was ["Box items", "Wrap package"...]
      "successors": "Delivery Initiation"    // Was ["Truck loaded", "Dispatch"...]
    }},
    {{
      "activity": "Handle Error",
      "predecessors": "System Failure",      // Was ["Crash", "Server Down"...]
      "successors": "Recovery Process"       // Was ["Reboot", "Restart"...]
    }}
  ]
}}

**Constraint:**
- Output **ONLY** the JSON object.
"""

SYSTEM_PROMPT_SYNONYM_STEP3 ="""
You are an expert Process Mining Analyst.
Your goal is to cluster activity labels into groups that represent the **Same Process Step**.

### INPUT DATA
You will receive objects with:
1. `activity`: The label.
2. `predecessors`: A summarized string (Input Context).
3. `successors`: A summarized string (Output Context).

### CLUSTERING LOGIC: FUZZY CONTEXT & SYNONYM BOOST
Compare pairs of activities (A and B). Decide if they are the same step based on two factors:

**FACTOR 1: CONTEXT SIMILARITY (The Base Rule)**
- Compare `predecessors_A` vs `predecessors_B` AND `successors_A` vs `successors_B`.
- **Do not look for exact string matches.**
- **Rule:** If the descriptions describe the **Same Business Phase** despite different wording, count it as a MATCH.

**FACTOR 2: LABEL SYNONYM BOOST (The Tie-Breaker)**
- **Rule:** If `activity_A` and `activity_B` are **Linguistic Synonyms** (e.g., "Verify" vs "Check"), you must be **MORE LENIENT** with context matching.
- **Logic:** "If labels imply the same action, allow minor deviations in context phrasing."

### FINAL DECISION MATRIX
1. **Contexts are Semantically Similar:** -> **GROUP**.
2. **Contexts have minor differences BUT Labels are Synonyms:** -> **GROUP** (Synonym Boost).
3. **Contexts are clearly different (Input or Output diverges):** -> **SEPARATE**.

### GLOBAL INSTRUCTION
- **Output:** A JSON object with a single key `"clusters"` containing a **List of Lists**.
- **Constraint:** Ensure Transitivity (A=B, B=C -> A=B=C).
"""

USER_PROMPT_SYNONYM_STEP3 = """
### TASK: Fuzzy Context Clustering with Synonym Boost

**OBJECTIVE:**
Group the activities in **INPUT DATA** that represent the same process step.
**Key Instruction:** Be flexible with context descriptions. Focus on the **Core Meaning**.

**STRICT EXECUTION STEPS:**

1. **Analyze Contexts:**
   - Read the natural language summaries.
   - Interpret different phrases describing the same stage as the **SAME** context (e.g., "Data Entry" ≈ "Inputting Data").

2. **Apply Clustering Logic (Use these Mental Models):**
   - **Case 1 (Strong Match):**
     - Context A: "User entering credentials"
     - Context B: "Inputting login details"
     - **Decision:** Contexts mean the same thing. -> **GROUP**.
   - **Case 2 (Synonym Boost):**
     - Labels: "Resolve Ticket" vs "Fix Issue" (Strong Synonyms).
     - Contexts: "Code review" vs "Peer review completed" (Slight wording diff).
     - **Decision:** Labels are synonyms, so ignore the slight context difference. -> **GROUP**.
   - **Case 3 (Mismatch):**
     - Labels: "Approve" vs "Reject".
     - Contexts: "Evaluation" vs "Evaluation". (Context match, but Labels opposite).
     - **Decision:** Clearly different outcome. -> **SEPARATE**.

3. **Apply Transitivity:**
   - Merge all overlapping pairs into final clusters.

**INPUT DATA (Summarized Context):**
{SYNONYM_STEP3_INPUT}

***OUTPUT FORMAT GUIDELINES***
Return a strict JSON Object with a single key `"clusters"`.

**Constraint:**
- Output **ONLY** the JSON object.
"""
