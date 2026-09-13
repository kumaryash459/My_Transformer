import torch

from training.batch_utils import prepare_decoder_inputs


target = torch.tensor([
    [1, 10, 20, 30, 2],
    [1, 15, 25, 35, 2]
])

decoder_input, expected_output = prepare_decoder_inputs(target)

print("Original:")
print(target)

print("\nDecoder input:")
print(decoder_input)

print("\nExpected output:")
print(expected_output)

assert torch.equal(
    decoder_input,
    torch.tensor([
        [1, 10, 20, 30],
        [1, 15, 25, 35]
    ])
)

assert torch.equal(
    expected_output,
    torch.tensor([
        [10, 20, 30, 2],
        [15, 25, 35, 2]
    ])
)

print("\nTarget shifting test passed.")