import torch

# number of entries
num_entries = 1000

db = torch.rand(num_entries) > 0.5  # big database of 1 and 0 with one column
print(db)

"""
If we remove a person from the database and the query doesn't change, that person's
privacy is fully protected
This means the person wasn't leaking any statistical information into the output of the
query
Can we construct a query which doesn't change no matter who we remove from the database?
"""

# Write a function that makes it so you can take this database and create 1000 other
# databases each with one person missing
