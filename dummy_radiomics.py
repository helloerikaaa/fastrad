import sys
from unittest.mock import MagicMock
sys.modules['radiomics'] = MagicMock()
sys.modules['radiomics.featureextractor'] = MagicMock()
