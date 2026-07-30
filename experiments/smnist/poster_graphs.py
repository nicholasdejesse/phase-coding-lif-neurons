import matplotlib.pyplot as plt
from textwrap import fill

# Data
models = ["Binary coding", "Delta modulation", "Phase-coded delta modulation"]
accuracies = [131754.84, 29021.96, 28591.40]

# Wrap labels at about 12 characters
wrapped_models = [fill(label, width=12) for label in models]

# Create bar chart
plt.figure(figsize=(6, 4))
plt.bar(wrapped_models, accuracies)

# Labels and formatting
plt.ylabel("Spike Operations")
plt.xlabel("Coding Scheme")
plt.title("Spike Operations by Coding Scheme")
plt.ylim(0, 145000)

offset = max(accuracies) * 0.015  # 2% of tallest bar


for i, acc in enumerate(accuracies):
    plt.text(i, acc + offset, f"{acc:.1f}", ha="center")

plt.tight_layout()
plt.savefig("comparison_spike.svg")
plt.show()