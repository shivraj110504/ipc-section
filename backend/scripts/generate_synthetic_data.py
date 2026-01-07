# build synthetic training data
import pandas as pd

IPC_ACTIONS = {
    "378": [
        "took movable property without consent",
        "stole a wallet from the victim",
        "took a mobile phone dishonestly"
    ],
    "420": [
        "cheated the victim in a financial transaction",
        "dishonestly induced delivery of property"
    ],
    "323": [
        "voluntarily caused hurt",
        "physically assaulted the complainant"
    ],
    "326": [
        "caused grievous injury using a weapon",
        "assaulted the victim with a knife"
    ],
    "506": [
        "threatened the victim with injury",
        "issued criminal intimidation"
    ]
}

TEMPLATES = [
    "The accused {action}.",
    "It is alleged that the accused {action}.",
    "According to the complaint, the accused {action}.",
    "The victim stated that the accused {action}.",
    "As per FIR records, the accused {action}."
]

rows = []

for ipc, actions in IPC_ACTIONS.items():
    for action in actions:
        for template in TEMPLATES:
            rows.append({
                "case_text": template.format(action=action),
                "ipc_sections": ipc
            })

df = pd.DataFrame(rows)

OUT_CSV = "backend/data/processed/ipc_training_data_synthetic.csv"
df.to_csv(OUT_CSV, index=False)

print(f"Training dataset created with {len(df)} samples")
