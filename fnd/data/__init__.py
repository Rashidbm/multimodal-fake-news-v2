"""Dataset construction package.

Pipeline (see docs/DATASET.md for the full explanation):

    raw datasets  --loaders.py-->  list[Sample] with a scenario 1..5
                  --build.py---->  balanced, de-duplicated, split CSV + manifest
                  --verify.py--->  independent re-check of every invariant
"""
