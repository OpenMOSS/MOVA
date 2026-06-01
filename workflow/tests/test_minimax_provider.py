"""Unit tests for MiniMax LLM provider integration."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add workflow directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestResolveApiKeys(unittest.TestCase):
    """Tests for api_utils.resolve_api_keys with MiniMax support."""

    def test_returns_three_tuple(self):
        from api_utils import resolve_api_keys
        result = resolve_api_keys()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_params_take_priority_over_env(self):
        from api_utils import resolve_api_keys
        with patch.dict(os.environ, {
            'GEMINI_API_KEY': 'env_gemini',
            'DASHSCOPE_API_KEY': 'env_qwen',
            'MINIMAX_API_KEY': 'env_minimax',
        }):
            gemini, qwen, minimax = resolve_api_keys(
                api_key='param_gemini',
                qwen_api_key='param_qwen',
                minimax_api_key='param_minimax',
            )
            self.assertEqual(gemini, 'param_gemini')
            self.assertEqual(qwen, 'param_qwen')
            self.assertEqual(minimax, 'param_minimax')

    def test_env_fallback(self):
        from api_utils import resolve_api_keys
        with patch.dict(os.environ, {
            'GEMINI_API_KEY': 'env_gemini',
            'DASHSCOPE_API_KEY': 'env_qwen',
            'MINIMAX_API_KEY': 'env_minimax',
        }):
            gemini, qwen, minimax = resolve_api_keys()
            self.assertEqual(gemini, 'env_gemini')
            self.assertEqual(qwen, 'env_qwen')
            self.assertEqual(minimax, 'env_minimax')

    def test_empty_when_no_keys(self):
        from api_utils import resolve_api_keys
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing env vars
            for key in ['GEMINI_API_KEY', 'DASHSCOPE_API_KEY', 'MINIMAX_API_KEY']:
                os.environ.pop(key, None)
            gemini, qwen, minimax = resolve_api_keys()
            self.assertEqual(gemini, '')
            self.assertEqual(qwen, '')
            self.assertEqual(minimax, '')

    def test_strips_whitespace(self):
        from api_utils import resolve_api_keys
        gemini, qwen, minimax = resolve_api_keys(
            api_key='  gemini  ',
            qwen_api_key='  qwen  ',
            minimax_api_key='  minimax  ',
        )
        self.assertEqual(gemini, 'gemini')
        self.assertEqual(qwen, 'qwen')
        self.assertEqual(minimax, 'minimax')

    def test_none_params_fall_through_to_env(self):
        from api_utils import resolve_api_keys
        with patch.dict(os.environ, {'MINIMAX_API_KEY': 'from_env'}):
            _, _, minimax = resolve_api_keys(minimax_api_key=None)
            self.assertEqual(minimax, 'from_env')


class TestMiniMaxConfig(unittest.TestCase):
    """Tests for MiniMax configuration in config.py."""

    def test_minimax_config_defaults(self):
        from config import MINIMAX_API_KEY, MINIMAX_BASE_URL, MINIMAX_MODEL
        self.assertIsInstance(MINIMAX_API_KEY, str)
        self.assertEqual(MINIMAX_BASE_URL, 'https://api.minimax.io/v1')
        self.assertEqual(MINIMAX_MODEL, 'MiniMax-M3')

    def test_minimax_config_from_env(self):
        with patch.dict(os.environ, {
            'MINIMAX_API_KEY': 'test_key_123',
            'MINIMAX_BASE_URL': 'https://custom.api/v1',
            'MINIMAX_MODEL': 'MiniMax-M2.7',
        }):
            # Re-import to pick up env vars
            import importlib
            import config
            importlib.reload(config)
            self.assertEqual(config.MINIMAX_API_KEY, 'test_key_123')
            self.assertEqual(config.MINIMAX_BASE_URL, 'https://custom.api/v1')
            self.assertEqual(config.MINIMAX_MODEL, 'MiniMax-M2.7')
            # Reload with clean env
            for key in ['MINIMAX_API_KEY', 'MINIMAX_BASE_URL', 'MINIMAX_MODEL']:
                os.environ.pop(key, None)
            importlib.reload(config)


class TestVideoDescriptionMinimax(unittest.TestCase):
    """Tests for generate_video_description_minimax in prompt_rewriter_with_image.py."""

    @patch('prompt_rewriter_with_image._OpenAI')
    def test_calls_openai_compatible_api(self, mock_openai_cls):
        from prompt_rewriter_with_image import generate_video_description_minimax

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A detailed video description..."
        mock_client.chat.completions.create.return_value = mock_response

        result = generate_video_description_minimax(
            user_input="A cat plays piano",
            first_frame_elements="A tabby cat sits at a piano",
            api_key="test_key",
            model="MiniMax-M2.7",
        )

        mock_openai_cls.assert_called_once_with(
            api_key="test_key",
            base_url="https://api.minimax.io/v1",
        )
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args.kwargs['model'], 'MiniMax-M2.7')
        self.assertEqual(len(call_args.kwargs['messages']), 2)
        self.assertEqual(call_args.kwargs['messages'][0]['role'], 'system')
        self.assertEqual(call_args.kwargs['messages'][1]['role'], 'user')
        self.assertIn('cat plays piano', call_args.kwargs['messages'][1]['content'])
        self.assertLessEqual(call_args.kwargs['temperature'], 1.0)
        self.assertGreater(call_args.kwargs['temperature'], 0.0)
        self.assertEqual(result, "A detailed video description...")

    @patch('prompt_rewriter_with_image._OpenAI')
    def test_custom_base_url(self, mock_openai_cls):
        from prompt_rewriter_with_image import generate_video_description_minimax

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "description"
        mock_client.chat.completions.create.return_value = mock_response

        generate_video_description_minimax(
            user_input="test",
            first_frame_elements="test",
            api_key="key",
            base_url="https://custom.api/v1",
        )

        mock_openai_cls.assert_called_once_with(
            api_key="key",
            base_url="https://custom.api/v1",
        )

    @patch('prompt_rewriter_with_image._OpenAI')
    def test_raises_on_empty_response(self, mock_openai_cls):
        from prompt_rewriter_with_image import generate_video_description_minimax

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_client.chat.completions.create.return_value = mock_response

        with self.assertRaises(ValueError):
            generate_video_description_minimax(
                user_input="test",
                first_frame_elements="test",
                api_key="key",
            )

    @patch('prompt_rewriter_with_image._OpenAI')
    def test_raises_on_none_response(self, mock_openai_cls):
        from prompt_rewriter_with_image import generate_video_description_minimax

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response

        with self.assertRaises(ValueError):
            generate_video_description_minimax(
                user_input="test",
                first_frame_elements="test",
                api_key="key",
            )

    @patch('prompt_rewriter_with_image._OpenAI')
    def test_strips_whitespace_from_result(self, mock_openai_cls):
        from prompt_rewriter_with_image import generate_video_description_minimax

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  result text  \n"
        mock_client.chat.completions.create.return_value = mock_response

        result = generate_video_description_minimax(
            user_input="test",
            first_frame_elements="test",
            api_key="key",
        )
        self.assertEqual(result, "result text")

    def test_raises_without_openai(self):
        import prompt_rewriter_with_image as module
        original = module._OpenAI
        module._OpenAI = None
        try:
            with self.assertRaises(ImportError):
                module.generate_video_description_minimax(
                    user_input="test",
                    first_frame_elements="test",
                    api_key="key",
                )
        finally:
            module._OpenAI = original


class TestImagePromptMinimax(unittest.TestCase):
    """Tests for generate_image_prompt_minimax in generate_first_frame.py."""

    @patch('generate_first_frame._OpenAI')
    def test_calls_openai_compatible_api(self, mock_openai_cls):
        from generate_first_frame import generate_image_prompt_minimax

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A medium shot of a cat at a grand piano..."
        mock_client.chat.completions.create.return_value = mock_response

        result = generate_image_prompt_minimax(
            user_input="A cat plays piano",
            api_key="test_key",
            model="MiniMax-M2.7",
        )

        mock_openai_cls.assert_called_once_with(
            api_key="test_key",
            base_url="https://api.minimax.io/v1",
        )
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args.kwargs['model'], 'MiniMax-M2.7')
        self.assertEqual(len(call_args.kwargs['messages']), 2)
        self.assertIn('cat plays piano', call_args.kwargs['messages'][1]['content'])
        self.assertEqual(result, "A medium shot of a cat at a grand piano...")

    @patch('generate_first_frame._OpenAI')
    def test_raises_on_empty_response(self, mock_openai_cls):
        from generate_first_frame import generate_image_prompt_minimax

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_client.chat.completions.create.return_value = mock_response

        with self.assertRaises(ValueError):
            generate_image_prompt_minimax(
                user_input="test",
                api_key="key",
            )

    @patch('generate_first_frame._OpenAI')
    def test_custom_base_url(self, mock_openai_cls):
        from generate_first_frame import generate_image_prompt_minimax

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "prompt"
        mock_client.chat.completions.create.return_value = mock_response

        generate_image_prompt_minimax(
            user_input="test",
            api_key="key",
            base_url="https://custom.api/v1",
        )

        mock_openai_cls.assert_called_once_with(
            api_key="key",
            base_url="https://custom.api/v1",
        )

    def test_raises_without_openai(self):
        import generate_first_frame as module
        original = module._OpenAI
        module._OpenAI = None
        try:
            with self.assertRaises(ImportError):
                module.generate_image_prompt_minimax(
                    user_input="test",
                    api_key="key",
                )
        finally:
            module._OpenAI = original

    @patch('generate_first_frame._OpenAI')
    def test_temperature_within_bounds(self, mock_openai_cls):
        from generate_first_frame import generate_image_prompt_minimax

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "prompt text"
        mock_client.chat.completions.create.return_value = mock_response

        generate_image_prompt_minimax(user_input="test", api_key="key")

        call_args = mock_client.chat.completions.create.call_args
        temp = call_args.kwargs['temperature']
        self.assertGreater(temp, 0.0)
        self.assertLessEqual(temp, 1.0)


class TestProviderFallbackChain(unittest.TestCase):
    """Tests for the Gemini → Qwen → MiniMax fallback chain."""

    @patch('prompt_rewriter_with_image._OpenAI')
    @patch('prompt_rewriter_with_image.genai', None)
    @patch('prompt_rewriter_with_image.dashscope', None)
    def test_minimax_fallback_in_prompt_rewriter(self, mock_openai_cls):
        """When only MiniMax key is set, prompt rewriter should use MiniMax."""
        from prompt_rewriter_with_image import (
            generate_video_description_minimax,
        )

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "MiniMax generated description"
        mock_client.chat.completions.create.return_value = mock_response

        result = generate_video_description_minimax(
            user_input="test input",
            first_frame_elements="test elements",
            api_key="minimax_key",
        )
        self.assertEqual(result, "MiniMax generated description")
        mock_openai_cls.assert_called_once()


class TestSystemPromptContent(unittest.TestCase):
    """Tests that system prompts are properly passed to MiniMax."""

    @patch('prompt_rewriter_with_image._OpenAI')
    def test_system_prompt_is_video_description_expert(self, mock_openai_cls):
        from prompt_rewriter_with_image import (
            generate_video_description_minimax,
            SYSTEM_INSTRUCTION,
        )

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "description"
        mock_client.chat.completions.create.return_value = mock_response

        generate_video_description_minimax(
            user_input="test",
            first_frame_elements="test",
            api_key="key",
        )

        call_args = mock_client.chat.completions.create.call_args
        system_msg = call_args.kwargs['messages'][0]
        self.assertEqual(system_msg['role'], 'system')
        self.assertEqual(system_msg['content'], SYSTEM_INSTRUCTION)

    @patch('generate_first_frame._OpenAI')
    def test_image_prompt_system_is_first_frame_expert(self, mock_openai_cls):
        from generate_first_frame import (
            generate_image_prompt_minimax,
            IMAGE_PROMPT_SYSTEM,
        )

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "prompt"
        mock_client.chat.completions.create.return_value = mock_response

        generate_image_prompt_minimax(user_input="test", api_key="key")

        call_args = mock_client.chat.completions.create.call_args
        system_msg = call_args.kwargs['messages'][0]
        self.assertEqual(system_msg['role'], 'system')
        self.assertEqual(system_msg['content'], IMAGE_PROMPT_SYSTEM)


if __name__ == '__main__':
    unittest.main()
