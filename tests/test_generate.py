from types import SimpleNamespace
from unittest.mock import patch

import torch

import generate as llada_generate


EOS_TOKEN_ID = 126081
EOT_TOKEN_ID = 126348
MASK_TOKEN_ID = 126336
VOCAB_SIZE = EOT_TOKEN_ID + 1


class _OrderSensitiveModel:
    device = torch.device("cpu")
    config = SimpleNamespace(_name_or_path="GSAI-ML/LLaDA-8B-Instruct")

    def __call__(self, input_ids, attention_mask=None):
        logits = torch.full((*input_ids.shape, VOCAB_SIZE), -20.0)
        first_masked = input_ids[0, 1].item() == MASK_TOKEN_ID
        second_masked = input_ids[0, 2].item() == MASK_TOKEN_ID

        if first_masked:
            if second_masked:
                logits[0, 1, EOS_TOKEN_ID] = 10.0
                logits[0, 1, 5] = 0.0
            else:
                logits[0, 1, 11] = 10.0

        if second_masked:
            if first_masked:
                logits[0, 2, 7] = 1.0
                logits[0, 2, 8] = 1.0
            else:
                logits[0, 2, 12] = 10.0

        return SimpleNamespace(logits=logits)


def test_eos_confidence_is_suppressed_after_sampling():
    model = _OrderSensitiveModel()
    prompt = torch.tensor([[42]])

    # Keep the nonzero-temperature tensor-copy semantics deterministic. The
    # sampler must still choose EOS before its confidence is suppressed.
    with patch.object(
        llada_generate,
        "add_gumbel_noise",
        side_effect=lambda logits, temperature: logits.clone(),
    ):
        output = llada_generate.generate(
            model,
            prompt,
            steps=2,
            gen_length=2,
            block_length=2,
            temperature=1.0,
            confidence_eos_eot_inf=True,
        )

    # The regular token is transferred first. The model then revises the
    # still-masked EOS candidate on the next denoising step.
    assert output.tolist() == [[42, 11, 7]]
