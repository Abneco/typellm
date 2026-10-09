# Notebooks

Paste your API keys in the cell near the top and run all cells: each notebook calls the APIs itself. card_suit_bets.ipynb, discover_invoice_categories.ipynb and hierarchical_classification.ipynb keep the outputs of one run, so you can read them before running them.

| Notebook | What it shows | Keys to paste |
| --- | --- | --- |
| [card_suit_bets.ipynb](card_suit_bets.ipynb) | Guess the suit of a random card: TypeLLM and OpenAI's Decisions API give a probability for each suit, then bet against each other. The side closer to the true 1 in 4 wins the pool. | `TYPELLM_API_KEY`, `OPENAI_API_KEY` |
| [discover_invoice_categories.ipynb](discover_invoice_categories.ipynb) | Sort invoices into categories that start from two: each call picks a category or names a new one. Keeps the outputs of one run. | `TYPELLM_API_KEY` |
| [hierarchical_classification.ipynb](hierarchical_classification.ipynb) | Classify products into a tree of 64 categories: pick a department, then a category inside it, in one call with `when`. Keeps the outputs of one run. | `TYPELLM_API_KEY` |
| [identical_resumes.ipynb](identical_resumes.ipynb) | Two candidates with the same résumé, asked in both orders: the fair answer is 50/50 whichever is listed first. Compares TypeLLM, Jev and OpenAI's Decisions API. | `TYPELLM_API_KEY`, `TYPESAFE_API_KEY`, `OPENAI_API_KEY` |
