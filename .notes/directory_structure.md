lex-db/
├── db/
│   └── lex.db                # SQLite database file
├── lex_db/                   # Main package
│   ├── __init__.py
│   ├── config.py             # Configuration settings
│   ├── database.py           # Database connection and operations
│   └── api/                  # API endpoints
│       ├── __init__.py
│       └── routes.py         # API routes
├── main.py                   # Entry point
├── .env                      # Environment variables
├── pyproject.toml            # Project metadata and dependencies
└── README.md                 # This file