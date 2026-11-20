import os
import unittest
from unittest.mock import patch

import yaml

from shared.settings import AdminSettings, DEFAULTS_PATH, GeneralSettings


class SharedConfigTests(unittest.TestCase):
    def test_yaml_keys_match_general_settings_schema(self) -> None:
        with DEFAULTS_PATH.open(encoding="utf-8") as defaults_file:
            defaults = yaml.safe_load(defaults_file)

        self.assertEqual(set(defaults), set(GeneralSettings.model_fields))

    def test_api_and_admin_share_yaml_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            general = GeneralSettings(_env_file=None)
            admin = AdminSettings(_env_file=None)

        self.assertEqual(general.COLLECTION_NAME, "financial_rag_analytics")
        self.assertEqual(admin.COLLECTION_NAME, general.COLLECTION_NAME)
        self.assertEqual(admin.EMBEDDING_MODEL, general.EMBEDDING_MODEL)
        self.assertEqual(admin.VECTOR_NAME, general.VECTOR_NAME)

    def test_environment_overrides_shared_yaml_defaults(self) -> None:
        with patch.dict(os.environ, {"COLLECTION_NAME": "team_collection"}, clear=True):
            general = GeneralSettings(_env_file=None)
            admin = AdminSettings(_env_file=None)

        self.assertEqual(general.COLLECTION_NAME, "team_collection")
        self.assertEqual(admin.COLLECTION_NAME, general.COLLECTION_NAME)
