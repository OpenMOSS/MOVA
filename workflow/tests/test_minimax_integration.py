"""Integration tests for MiniMax LLM provider (require MINIMAX_API_KEY)."""

import os
import sys
import unittest

# Add workflow directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

MINIMAX_API_KEY = os.environ.get('MINIMAX_API_KEY', '')


@unittest.skipUnless(MINIMAX_API_KEY, 'MINIMAX_API_KEY not set')
class TestMiniMaxVideoDescriptionIntegration(unittest.TestCase):
    """Integration tests for video description generation with MiniMax API."""

    def test_generate_video_description(self):
        from prompt_rewriter_with_image import generate_video_description_minimax

        result = generate_video_description_minimax(
            user_input="A cat sits at a grand piano in a cozy living room and plays a gentle melody",
            first_frame_elements=(
                "Visual Style: Warm cinematic realism. "
                "Camera: Medium shot at eye level. "
                "Visual Elements: A tabby cat sits on a piano bench facing a black grand piano. "
                "Warm ambient lighting from a floor lamp."
            ),
            api_key=MINIMAX_API_KEY,
        )

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 50)

    def test_generate_video_description_m27_highspeed(self):
        from prompt_rewriter_with_image import generate_video_description_minimax

        result = generate_video_description_minimax(
            user_input="A dog runs through a field of sunflowers on a sunny day",
            first_frame_elements=(
                "Visual Style: Bright outdoor photography. "
                "Camera: Wide shot. "
                "Visual Elements: A golden retriever mid-stride in a sunflower field."
            ),
            api_key=MINIMAX_API_KEY,
            model="MiniMax-M2.7-highspeed",
        )

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 50)


@unittest.skipUnless(MINIMAX_API_KEY, 'MINIMAX_API_KEY not set')
class TestMiniMaxImagePromptIntegration(unittest.TestCase):
    """Integration tests for image prompt generation with MiniMax API."""

    def test_generate_image_prompt(self):
        from generate_first_frame import generate_image_prompt_minimax

        result = generate_image_prompt_minimax(
            user_input="A cozy coffee shop on a rainy evening with warm lighting inside",
            api_key=MINIMAX_API_KEY,
        )

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 30)


if __name__ == '__main__':
    unittest.main()
