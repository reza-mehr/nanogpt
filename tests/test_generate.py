
'''
This file contains code to test the generation utilities and functions.

'''

#%% Imports:

from nanogpt.generation import truncate_at_end

#%% Tests:

def test_truncate_at_end():
    assert truncate_at_end([1, 2, 9, 3, 9], end_id=9, start=0) == [1, 2]
    assert truncate_at_end([9, 2, 3, 9, 4], end_id=9, start=1) == [9, 2, 3]   # skips the prompt
    assert truncate_at_end([1, 2, 3], end_id=9, start=0) == [1, 2, 3]         # no end token
    assert truncate_at_end([1, 0, 2], end_id=0, start=0) == [1]               # ID 0 is valid
    assert truncate_at_end([1, 9, 2], end_id=None, start=0) == [1, 9, 2]      # no stop configured
