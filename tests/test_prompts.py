import unittest

from app.utils.guardrails import _load_rails_config
from shared.prompts import load_prompt, render_prompt


class PromptTests(unittest.TestCase):
    def test_chat_prompt_template_is_rendered(self) -> None:
        rendered = render_prompt(
            "chat.yaml",
            "rag_user_with_focus",
            query="Revenue for $5",
            search_query="revenue",
        )

        self.assertEqual(rendered, "Revenue for $5\n(Retrieval focus: revenue)")

    def test_router_prompt_preserves_json_schema_and_substitutes_values(self) -> None:
        rendered = render_prompt(
            "router.yaml",
            "query_router",
            known_companies='["Example Co"]',
            session_state='{"company": null}',
            history="No prior history",
            query="What about $5?",
        )

        self.assertIn('{"intent": "rag_search"', rendered)
        self.assertIn('Latest User Query: "What about $5?"', rendered)

    def test_missing_template_value_fails_explicitly(self) -> None:
        with self.assertRaises(KeyError):
            render_prompt("router.yaml", "query_router", query="Revenue?")

    def test_guardrails_prompt_uses_shared_yaml(self) -> None:
        prompt = load_prompt("guardrails.yaml", "self_check_input")

        self.assertIn("User message: \"{{ user_input }}\"", prompt)

    def test_nemo_config_loads_shared_guardrails_prompt(self) -> None:
        config = _load_rails_config()
        prompts = [prompt for prompt in config.prompts if prompt.task == "self_check_input"]

        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0].content, load_prompt("guardrails.yaml", "self_check_input"))
        self.assertTrue(config.flows)

        from nemoguardrails.colang.runtime import Runtime

        runtime = Runtime(config)
        self.assertIsNotNone(runtime.action_dispatcher.get_action("check_input_heuristics"))
        self.assertIsNotNone(runtime.action_dispatcher.get_action("check_numeric_grounding"))
