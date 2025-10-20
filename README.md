# 🧠 Sparse Mixture-of-Experts Transformer (Noisy Top-k Routing LLM)

This project implements a lightweight, modular version of a **Mixture-of-Experts (MoE)** Transformer with **Noisy Top-k routing**, inspired by architectures such as **Switch Transformer (Google)**, **GLaM**, and **Mixtral (Mistral AI)** — implemented fully from scratch in PyTorch.

---

## 🚀 Overview

This model combines the efficiency of sparse expert activation with the expressive power of Transformers.

- **Sparse Expert Routing** — activates only `top_k` experts per token, making the model efficient.  
- **Noisy Top-k Gating** — adds Gaussian noise for balanced expert utilization (like Switch Transformer).  
- **Multi-Head Self-Attention** — handles contextual communication between tokens.  
- **Mixture-of-Experts Feedforward Layer** — distributes computation across multiple specialized sub-networks.  
- **End-to-End Implementation** — includes data loading, batching, training, evaluation, and inference.

---

## 🧩 Architecture Overview



Each block alternates between **attention** (context mixing) and **MoE experts** (computation).  
The **NoisyTopkRouter** decides which experts are active for each token dynamically.

---

## ⚙️ Hyperparameters

| Parameter | Value | Description |
|------------|--------|-------------|
| `batch_size` | 16 | sequences processed in parallel |
| `block_size` | 32 | context length for predictions |
| `n_embed` | 128 | embedding dimension |
| `n_head` | 8 | number of attention heads |
| `n_layer` | 8 | number of Transformer blocks |
| `num_experts` | 8 | total experts in the MoE layer |
| `top_k` | 2 | active experts per token |
| `dropout` | 0.1 | dropout regularization |
| `learning_rate` | 1e-3 | optimizer learning rate |
| `device` | auto | CUDA / CPU |
| `max_iters` | 200 | training iterations (for quick demo) |

🧠 *This notebook was run for **200 iterations** to demonstrate the concept end-to-end.*  
You can train much longer for better results (see below).

---

## 🧮 Training Loop Example

```python
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



🧪 Experimental Notes

This project was designed as a research and experimentation notebook.
If you want to make it fully end-to-end or achieve higher accuracy:

🔁 Increase training iterations: Run between 60,000–100,000 iterations if you have a high-end GPU (e.g., RTX 4090, A100, or H100).

⚙️ Tune hyperparameters: Experiment with learning_rate, num_experts, top_k, and n_embed.

💾 Save and share weights: Push trained weights to Hugging Face Hub
 for public usage.

💬 Create an interactive interface: Use Gradio or Streamlit to interact with your trained model in real time.

🧠 Explore specialization: Visualize which experts activate for which tokens (expert heatmaps).
