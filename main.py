from transformers import pipeline

# ====================================
# 1. Meeting Transcript
# ====================================

transcript = """

The team discussed the hiring plan for the next quarter.

We need to recruit three new AI engineers.

Mithun will prepare the hiring report by Friday.

The budget for hiring will be increased by 20%.

Rahul will contact recruitment agencies.

The team agreed to purchase additional GPU servers.

"""

# ====================================
# 2. Split Into Sentences
# ====================================

sentences = [

    sentence.strip()

    for sentence in transcript.split(".")

    if sentence.strip()

]

# ====================================
# 3. Transformer Classifier
# ====================================

classifier = pipeline(

    "zero-shot-classification",

    model="facebook/bart-large-mnli"

)

# ====================================
# 4. Labels
# ====================================

topic_labels = [

    "Hiring",
    "Budget",
    "Infrastructure",
    "Marketing",
    "Sales"
]

# ====================================
# 5. Extract Topics
# ====================================

topics = []

for sentence in sentences:

    result = classifier(

        sentence,

        topic_labels

    )

    label = result["labels"][0]

    score = result["scores"][0]

    if score > 0.50:

        topics.append(label)

# ====================================
# 6. Action Items
# ====================================

action_items = []

for sentence in sentences:

    lower = sentence.lower()

    if (

        "will" in lower

        or

        "prepare" in lower

        or

        "contact" in lower

    ):

        action_items.append(

            sentence

        )

# ====================================
# 7. Decisions
# ====================================

decisions = []

for sentence in sentences:

    lower = sentence.lower()

    if (

        "agreed" in lower

        or

        "approved" in lower

        or

        "decided" in lower

    ):

        decisions.append(

            sentence

        )

# ====================================
# 8. Final Report
# ====================================

print("\nMEETING REPORT")

print("\nTopics:")

for topic in sorted(set(topics)):

    print("-", topic)

print("\nDecisions:")

for decision in decisions:

    print("-", decision)

print("\nAction Items:")

for action in action_items:

    print("-", action)
