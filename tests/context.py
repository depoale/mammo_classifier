"""Context used to import modules"""

import os
import sys
'''
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "codice"))
)
'''
print(os.getcwd())



print(os.getcwd())

os.chdir('codice')

print(os.getcwd())
