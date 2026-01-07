# IPC_ACTIONS (legal actions)
"""
IPC Action Library
------------------
Legally grounded action phrases derived from IPC definitions.
Used only for synthetic training data generation.
"""

IPC_ACTIONS = {
    "378": [  # Theft
        "took movable property without consent",
        "dishonestly removed a wallet belonging to the victim",
        "took a mobile phone secretly",
        "removed cash from another person's possession"
    ],
    "420": [  # Cheating
        "cheated the victim in a financial transaction",
        "dishonestly induced the victim to transfer property",
        "committed fraud for wrongful gain",
        "misrepresented facts to obtain money"
    ],
    "323": [  # Voluntarily causing hurt
        "voluntarily caused hurt to the victim",
        "physically assaulted the complainant",
        "caused bodily pain without grave injury"
    ],
    "326": [  # Grievous hurt with weapon
        "caused grievous injury using a sharp weapon",
        "assaulted the victim with a knife causing severe harm"
    ],
    "506": [  # Criminal intimidation
        "threatened the victim with injury",
        "issued criminal intimidation",
        "threatened to cause harm to the complainant"
    ],
    "354": [  # Outraging modesty
        "used criminal force against a woman",
        "outraged the modesty of a woman in public",
        "sexually harassed a woman"
    ]
}
