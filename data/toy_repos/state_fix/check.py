from state import update
original = {'a': 1}
assert update(original, 'b', 2) == {'a': 1, 'b': 2}
assert original == {'a': 1}
print('PASS')
