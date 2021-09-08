import importlib
from pathlib import Path

# export modules
for p in Path(__file__).parent.glob('*.py'):
    name = p.name.split('.py')[0]
    if name == '__init__':
        continue

    try:
        importlib.import_module('.' + name, 'modules.experts')
    except ModuleNotFoundError:
        pass
