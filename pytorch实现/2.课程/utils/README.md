使用格式

```python
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))  # .../utils
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)
```