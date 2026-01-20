# CoQA data utilities

Module: `immas.data.coqa.loader`

## CoQA dataset usage

```python
from datasets import load_dataset

ds = load_dataset("stanfordnlp/coqa")
```

## CoQA dataset format

- **source**: `wikipedia`

- **story**: The Vatican Apostolic Library (), more commonly called the Vatican Library or simply the Vat, is the library of the Holy See, located in Vatican City. Formally established in 1475, although it is much older, it is one of the oldest libraries in the world and contains one of the most significant collections of historical texts. It has 75,000 codices from throughout history, as well as 1.1 million printed books, which include some 8,500 incunabula. The Vatican Library is a research library for history, law, philosophy, science and theology. The Vatican Library is open to anyone who can document their qualifications and research needs. Photocopies for private study of pages from books published between 1801 and 1990 can be requested in person or by mail. In March 2014, the Vatican Library began an initial four-year project of digitising its collection of manuscripts, to be made available online. The Vatican Secret Archives were separated from the library at the beginning of the 17th century; they contain another 150,000 items. Scholars have traditionally divided the history of the library into five periods, Pre-Lateran, Lateran, Avignon, Pre-Vatican and Vatican. The Pre-Lateran period, comprising the initial days of the library, dated from the earliest days of the Church. Only a handful of volumes survive from this period, though some are very significant.

- **questions**:

  ```json
  [
    "When was the Vat formally opened?",
    "what is the library for?",
    "for what subjects?",
    "and?",
    "what was started in 2014?",
    "how do scholars divide the library?",
    "how many?",
    "what is the official name of the Vat?",
    "where is it?",
    "how many printed books does it contain?",
    "when were the Secret Archives moved from the rest of the library?",
    "how many items are in this secret collection?",
    "Can anyone use this library?",
    "what must be requested to view?",
    "what must be requested in person or by mail?",
    "of what books?",
    "What is the Vat the library of?",
    "How many books survived the Pre Lateran period?",
    "what is the point of the project started in 2014?",
    "what will this allow?"
  ]
  ```

- **answers**:

  ```json
  {
    "input_text": [
      "It was formally established in 1475",
      "research",
      "history, and law",
      "philosophy, science and theology",
      "a  project",
      "into periods",
      "five",
      "The Vatican Apostolic Library",
      "in Vatican City",
      "1.1 million",
      "at the beginning of the 17th century;",
      "150,000",
      "anyone who can document their qualifications and research needs.",
      "unknown",
      "Photocopies",
      "only books published between 1801 and 1990",
      "the Holy See",
      "a handful of volumes",
      "digitising manuscripts",
      "them to be viewed online."
    ],
    "answer_start": [
      151,
      454,
      457,
      457,
      769,
      1048,
      1048,
      4,
      94,
      328,
      917,
      915,
      546,
      -1,
      643,
      644,
      78,
      1192,
      785,
      868
    ],
    "answer_end": [
      179,
      494,
      511,
      545,
      879,
      1127,
      1128,
      94,
      150,
      412,
      1009,
      1046,
      643,
      -1,
      764,
      724,
      125,
      1384,
      881,
      910
    ]
  }
  ```

## `CoqaDialogue`

A normalized representation of one CoQA example:

- `dialogue_id: str`
- `source: str`
- `story: str`
- `questions: list[str]`
- `answers: list[str]`

Guarantee:

- `len(questions) == len(answers)` or construction fails.

## `CoqaDatasetIndex`

Optional convenience wrapper to load a full split and index by `dialogue_id`.

Main methods:

- `from_hf(split="validation")`
- `get_dialogue(dialogue_id)`
- `get_answer(dialogue_id, turn_number)` (1-based)
